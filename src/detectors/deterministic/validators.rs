//! Structural validators.
//!
//! Each is a pure function over a candidate string, independently testable and
//! independent of the pattern that produced the candidate. This split is the
//! whole point of the deterministic detector's design: patterns are cheap and
//! over-generate, validators are what buy precision.
//!
//! Where the language gives us a real parser (`Ipv4Addr`, `Ipv6Addr`), we use
//! it instead of a regex — a regex that accepts `999.999.999.999` is a
//! false-positive generator, and `str::parse` is both shorter and correct.

use std::net::{Ipv4Addr, Ipv6Addr};

/// Luhn mod-10 check over the ASCII digits in `s`, ignoring separators.
///
/// Rejects anything outside 12..=19 digits — the ISO/IEC 7812 PAN range. A
/// bare `is_luhn_valid` over arbitrary digit runs would flag roughly one in ten
/// random 16-digit numbers, which on a corpus of logs and build IDs is a
/// precision disaster.
pub fn luhn_valid(s: &str) -> bool {
    let digits: Vec<u8> = s
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| b - b'0')
        .collect();
    if !(12..=19).contains(&digits.len()) {
        return false;
    }
    // All-same-digit sequences (0000..., 1111...) pass Luhn surprisingly often
    // and are almost always test data or padding, not a real PAN.
    if digits.windows(2).all(|w| w[0] == w[1]) {
        return false;
    }

    let mut sum = 0u32;
    for (i, &d) in digits.iter().rev().enumerate() {
        let mut v = u32::from(d);
        if i % 2 == 1 {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
    }
    sum.is_multiple_of(10)
}

/// Card network implied by the BIN, used only to sanity-check length.
pub fn card_network(s: &str) -> Option<&'static str> {
    let d: String = s.chars().filter(char::is_ascii_digit).collect();
    let n = d.len();
    let two: u32 = d.get(0..2)?.parse().ok()?;
    let four: u32 = d.get(0..4).and_then(|x| x.parse().ok()).unwrap_or(0);

    Some(match () {
        _ if d.starts_with('4') && (13..=19).contains(&n) => "visa",
        _ if (51..=55).contains(&two) && n == 16 => "mastercard",
        _ if (2221..=2720).contains(&four) && n == 16 => "mastercard",
        _ if (two == 34 || two == 37) && n == 15 => "amex",
        _ if d.starts_with("6011") && n == 16 => "discover",
        _ if (644..=649).contains(&(four / 10)) && n == 16 => "discover",
        _ if d.starts_with("65") && n == 16 => "discover",
        _ if (3528..=3589).contains(&four) && (16..=19).contains(&n) => "jcb",
        _ if (two == 30 || two == 36 || two == 38) && (14..=19).contains(&n) => "diners",
        _ => return None,
    })
}

/// ISO 13616 IBAN validation: country length table plus the mod-97-10 checksum.
///
/// The length table matters. The checksum alone accepts about 1 in 97 random
/// strings of the right shape, which is far too loose for a benchmark that
/// scores false positives.
pub fn iban_valid(s: &str) -> bool {
    let compact: String = s
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if compact.len() < 15 || compact.len() > 34 {
        return false;
    }
    if !compact.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return false;
    }
    let cc = &compact[0..2];
    if !cc.bytes().all(|b| b.is_ascii_uppercase()) {
        return false;
    }
    if !compact[2..4].bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    match iban_length(cc) {
        Some(expected) if expected == compact.len() => {}
        _ => return false,
    }

    // Move the first four characters to the end, map letters to 10..35, then
    // take mod 97 incrementally to avoid big-integer arithmetic.
    let rearranged = format!("{}{}", &compact[4..], &compact[0..4]);
    let mut rem: u32 = 0;
    for c in rearranged.chars() {
        let v = if c.is_ascii_digit() {
            c as u32 - '0' as u32
        } else if c.is_ascii_uppercase() {
            c as u32 - 'A' as u32 + 10
        } else {
            return false;
        };
        rem = if v >= 10 {
            (rem * 100 + v) % 97
        } else {
            (rem * 10 + v) % 97
        };
    }
    rem == 1
}

/// Registered IBAN length by country code.
pub fn iban_length(cc: &str) -> Option<usize> {
    Some(match cc {
        "AD" => 24,
        "AE" => 23,
        "AL" => 28,
        "AT" => 20,
        "AZ" => 28,
        "BA" => 20,
        "BE" => 16,
        "BG" => 22,
        "BH" => 22,
        "BR" => 29,
        "BY" => 28,
        "CH" => 21,
        "CR" => 22,
        "CY" => 28,
        "CZ" => 24,
        "DE" => 22,
        "DK" => 18,
        "DO" => 28,
        "EE" => 20,
        "EG" => 29,
        "ES" => 24,
        "FI" => 18,
        "FO" => 18,
        "FR" => 27,
        "GB" => 22,
        "GE" => 22,
        "GI" => 23,
        "GL" => 18,
        "GR" => 27,
        "GT" => 28,
        "HR" => 21,
        "HU" => 28,
        "IE" => 22,
        "IL" => 23,
        "IQ" => 23,
        "IS" => 26,
        "IT" => 27,
        "JO" => 30,
        "KW" => 30,
        "KZ" => 20,
        "LB" => 28,
        "LC" => 32,
        "LI" => 21,
        "LT" => 20,
        "LU" => 20,
        "LV" => 21,
        "LY" => 25,
        "MC" => 27,
        "MD" => 24,
        "ME" => 22,
        "MK" => 19,
        "MR" => 27,
        "MT" => 31,
        "MU" => 30,
        "NL" => 18,
        "NO" => 15,
        "PK" => 24,
        "PL" => 28,
        "PS" => 29,
        "PT" => 25,
        "QA" => 29,
        "RO" => 24,
        "RS" => 22,
        "SA" => 24,
        "SC" => 31,
        "SE" => 24,
        "SI" => 19,
        "SK" => 24,
        "SM" => 27,
        "ST" => 25,
        "SV" => 28,
        "TL" => 23,
        "TN" => 24,
        "TR" => 26,
        "UA" => 29,
        "VA" => 22,
        "VG" => 24,
        "XK" => 20,
        _ => return None,
    })
}

/// Parse-based IPv4 validation. Rejects `999.1.1.1`, `1.2.3`, `01.2.3.4`.
pub fn ipv4_valid(s: &str) -> bool {
    // `Ipv4Addr::from_str` already rejects leading zeros and out-of-range
    // octets, which is precisely why we do not hand-roll this.
    s.parse::<Ipv4Addr>().is_ok()
}

/// Parse-based IPv6 validation.
pub fn ipv6_valid(s: &str) -> bool {
    s.parse::<Ipv6Addr>().is_ok()
}

/// Whether an IP is private, loopback, link-local or otherwise non-routable.
///
/// **Policy-dependent**, so it is a separate query rather than baked into
/// validation. `10.20.30.40` in a log line is meaningful to some DLP policies
/// and noise to others; the detector exposes it as a flag and the benchmark
/// records which setting produced a result.
pub fn ip_is_private(s: &str) -> bool {
    if let Ok(v4) = s.parse::<Ipv4Addr>() {
        return v4.is_private()
            || v4.is_loopback()
            || v4.is_link_local()
            || v4.is_unspecified()
            || v4.is_broadcast()
            || v4.is_documentation()
            || v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]); // CGNAT
    }
    if let Ok(v6) = s.parse::<Ipv6Addr>() {
        let seg = v6.segments();
        return v6.is_loopback()
            || v6.is_unspecified()
            || (seg[0] & 0xfe00) == 0xfc00      // unique local
            || (seg[0] & 0xffc0) == 0xfe80      // link local
            || seg[0] == 0x2001 && seg[1] == 0x0db8; // documentation
    }
    false
}

/// Structural JWT check: three base64url segments whose header decodes to JSON
/// containing `alg`.
///
/// The signature is deliberately not verified — we have no key, and an
/// unverifiable JWT is exactly as sensitive as a verifiable one.
pub fn jwt_valid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 {
        return false;
    }
    if parts[0].len() < 4 || parts[1].len() < 2 {
        return false;
    }
    if !parts.iter().all(|p| p.bytes().all(is_base64url_byte)) {
        return false;
    }
    // Signature may be empty only for `alg: none`, which we still treat as a
    // token; but a middle segment must exist.
    let Some(header) = base64url_decode(parts[0]) else {
        return false;
    };
    let Ok(text) = String::from_utf8(header) else {
        return false;
    };
    // A real header is a small JSON object naming an algorithm. Checking for
    // `alg` rather than just `{` rejects the dotted version strings and
    // package names that otherwise look base64url-ish.
    text.starts_with('{') && text.contains("\"alg\"")
}

fn is_base64url_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'='
}

/// Minimal unpadded base64url decoder. Returns `None` on invalid input.
pub fn base64url_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for b in s.bytes() {
        if b == b'=' {
            break;
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

/// AWS access key id: `AKIA`/`ASIA`/`ABIA`/`ACCA` followed by 16 uppercase
/// alphanumerics.
pub fn aws_access_key_valid(s: &str) -> bool {
    if s.len() != 20 {
        return false;
    }
    let prefix = &s[0..4];
    if !matches!(
        prefix,
        "AKIA" | "ASIA" | "ABIA" | "ACCA" | "AIDA" | "AROA" | "AIPA"
    ) {
        return false;
    }
    s[4..]
        .bytes()
        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

/// GitHub token: `ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`/`github_pat_` plus a
/// sufficiently long body.
pub fn github_token_valid(s: &str) -> bool {
    if let Some(rest) = s.strip_prefix("github_pat_") {
        return rest.len() >= 22 && rest.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    }
    let Some((prefix, rest)) = s.split_at_checked(4) else {
        return false;
    };
    matches!(prefix, "ghp_" | "gho_" | "ghu_" | "ghs_" | "ghr_")
        && rest.len() >= 36
        && rest.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// Shannon entropy in bits per character. Used to keep generic secret rules
/// from firing on prose.
pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    let mut n = 0usize;
    for b in s.bytes() {
        counts[b as usize] += 1;
        n += 1;
    }
    let n = n as f64;
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luhn_accepts_known_good_test_pans() {
        // The standard vendor test numbers.
        for pan in [
            "4111111111111111",    // Visa
            "4012888888881881",    // Visa
            "5500005555555559",    // Mastercard
            "5105105105105100",    // Mastercard
            "371449635398431",     // Amex
            "6011000991300009",    // Discover
            "3530111333300000",    // JCB
            "4111 1111 1111 1111", // separators ignored
            "4111-1111-1111-1111",
        ] {
            assert!(luhn_valid(pan), "{pan} should pass Luhn");
        }
    }

    #[test]
    fn luhn_rejects_bad_checksums_and_wrong_lengths() {
        for bad in [
            "4111111111111112", // last digit wrong
            "1234567890123456",
            "5500005555555558",
            "123456789012",         // 12 digits, bad checksum
            "411111111111",         // truncated
            "41111111111111111111", // 20 digits, out of PAN range
            "",
            "abcd",
        ] {
            assert!(!luhn_valid(bad), "{bad} should fail Luhn");
        }
    }

    #[test]
    fn luhn_rejects_repeated_digit_padding() {
        // Passes the arithmetic but is always padding or test data. This is a
        // real false-positive source in logs and fixtures.
        assert!(!luhn_valid("0000000000000000"));
        assert!(!luhn_valid("9999999999999999"));
    }

    #[test]
    fn card_network_checks_bin_and_length_together() {
        assert_eq!(card_network("4111111111111111"), Some("visa"));
        assert_eq!(card_network("5500005555555559"), Some("mastercard"));
        assert_eq!(card_network("2221000000000009"), Some("mastercard"));
        assert_eq!(card_network("371449635398431"), Some("amex"));
        assert_eq!(card_network("6011000991300009"), Some("discover"));
        // Right BIN, wrong length: Amex is 15 digits, not 16.
        assert_eq!(card_network("3714496353984311"), None);
        assert_eq!(card_network("1234567890123456"), None);
    }

    #[test]
    fn iban_accepts_registry_examples() {
        for iban in [
            "GB82 WEST 1234 5698 7654 32",
            "DE89370400440532013000",
            "FR1420041010050500013M02606",
            "NL91ABNA0417164300",
            "CH9300762011623852957",
            "IT60X0542811101000000123456",
            "ES9121000418450200051332",
            "BE68539007547034",
            "NO9386011117947",
            "de89370400440532013000", // case-insensitive
        ] {
            assert!(iban_valid(iban), "{iban} should be a valid IBAN");
        }
    }

    #[test]
    fn iban_rejects_bad_checksum_wrong_length_and_unknown_country() {
        for bad in [
            "GB82WEST12345698765433", // checksum broken
            "DE89370400440532013001",
            "DE8937040044053201300",   // 21 chars, DE needs 22
            "DE893704004405320130000", // 23 chars
            "ZZ89370400440532013000",  // unknown country
            "GB82",
            "1234567890123456", // no country code
            "",
        ] {
            assert!(!iban_valid(bad), "{bad} should be rejected");
        }
    }

    #[test]
    fn ip_validation_uses_the_real_parser() {
        assert!(ipv4_valid("192.0.2.1"));
        assert!(ipv4_valid("255.255.255.255"));
        // Each of these is accepted by a naive `\d{1,3}(\.\d{1,3}){3}` regex.
        assert!(!ipv4_valid("999.1.1.1"));
        assert!(!ipv4_valid("256.1.1.1"));
        assert!(!ipv4_valid("1.2.3"));
        assert!(!ipv4_valid("1.2.3.4.5"));
        assert!(!ipv4_valid("01.2.3.4"));
        assert!(!ipv4_valid("1.2.3."));

        assert!(ipv6_valid("2001:db8::1"));
        assert!(ipv6_valid("::1"));
        assert!(ipv6_valid("fe80::1ff:fe23:4567:890a"));
        assert!(!ipv6_valid("2001:db8:::1"));
        assert!(!ipv6_valid("gggg::1"));
    }

    #[test]
    fn private_ip_classification_is_explicit() {
        for p in [
            "10.20.30.40",
            "192.168.1.1",
            "172.16.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "0.0.0.0",
            "100.64.0.1",
            "::1",
            "fe80::1",
            "fd00::1",
            "2001:db8::1",
        ] {
            assert!(
                ip_is_private(p),
                "{p} should be classified private/non-routable"
            );
        }
        for p in ["8.8.8.8", "1.1.1.1", "172.32.0.1", "2606:4700::1111"] {
            assert!(!ip_is_private(p), "{p} should be classified public");
        }
    }

    #[test]
    fn jwt_requires_a_decodable_header_naming_an_algorithm() {
        // {"alg":"HS256","typ":"JWT"} . {"sub":"1234567890"} . sig
        let good = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.\
                    eyJzdWIiOiIxMjM0NTY3ODkwIn0.\
                    dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert!(jwt_valid(good));

        for bad in [
            "notbase64.notbase64.notbase64",
            "eyJhbGciOiJIUzI1NiJ9",         // two segments missing
            "eyJhbGciOiJIUzI1NiJ9.payload", // only two segments
            "1.23.4567",                    // a version number
            "a.b.c",
            // Valid base64url, but the header is not a JWT header.
            "aGVsbG8gd29ybGQ.eyJzdWIiOjF9.sig",
        ] {
            assert!(!jwt_valid(bad), "{bad} should not be a JWT");
        }
    }

    #[test]
    fn base64url_decoder_round_trips_and_rejects_junk() {
        assert_eq!(base64url_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64url_decode("").unwrap(), b"");
        assert!(base64url_decode("has space").is_none());
        assert!(
            base64url_decode("plus+slash/").is_none(),
            "standard base64 is not base64url"
        );
    }

    #[test]
    fn aws_and_github_token_shapes() {
        assert!(aws_access_key_valid("AKIAIOSFODNN7EXAMPLE"));
        assert!(aws_access_key_valid("ASIAY34FZKBOKMUTVV7A"));
        assert!(!aws_access_key_valid("AKIAIOSFODNN7EXAMPL"), "19 chars");
        assert!(
            !aws_access_key_valid("XXXXIOSFODNN7EXAMPLE"),
            "wrong prefix"
        );
        assert!(
            !aws_access_key_valid("akiaiosfodnn7example"),
            "must be uppercase"
        );

        assert!(github_token_valid(&format!("ghp_{}", "a".repeat(36))));
        assert!(github_token_valid(&format!(
            "github_pat_{}",
            "A1_".repeat(10)
        )));
        assert!(
            !github_token_valid(&format!("ghp_{}", "a".repeat(10))),
            "too short"
        );
        assert!(!github_token_valid("ghp_"), "empty body");
        assert!(
            !github_token_valid("ghx_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "bad prefix"
        );
    }

    #[test]
    fn entropy_separates_prose_from_random_strings() {
        let prose = shannon_entropy("the quick brown fox jumps over the lazy dog");
        let random = shannon_entropy("k3J8xQ2vB9nM4pL7wR1tY6zA5cF0dS8g");
        assert!(
            random > prose,
            "random={random} should exceed prose={prose}"
        );
        assert!(random > 4.0, "got {random}");
        assert_eq!(shannon_entropy(""), 0.0);
        assert_eq!(shannon_entropy("aaaa"), 0.0);
    }
}
