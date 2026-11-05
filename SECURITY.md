# Security policy

Please do not disclose vulnerabilities in a public issue. Use GitHub's private vulnerability reporting feature for this repository.

Do not commit access tokens, private model credentials, proprietary corpora, personally identifiable information, or unredacted runtime traces. Treat third-party model and dataset inputs as untrusted, and validate paths and serialized calibration artifacts before use.

TRIDENT can load Hugging Face models by identifier or from a local path. Use
trusted local weights or pin remote model revisions to immutable commits;
mutable model identifiers are outside TRIDENT's trust boundary. Dense indexes
use non-executable NumPy archives and should still be accepted only from
trusted sources.

