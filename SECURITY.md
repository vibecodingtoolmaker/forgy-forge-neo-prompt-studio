# Security Policy

## Supported versions

Forgy — Forge Neo Prompt Studio is currently a beta project. Security fixes are
provided on a best-effort basis for the latest published beta only.

| Version | Supported |
|---|---|
| Latest `0.5.x` beta | Yes |
| Older versions | No |

Before reporting a problem, reproduce it with the latest release when that can
be done safely.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability and do not include
secrets, private prompts, personas, generated images, model files, or sensitive
logs in a report.

Use **Report a vulnerability** on the repository's **Security** tab. GitHub's
private vulnerability reporting flow keeps the report and follow-up discussion
visible only to the reporter and repository maintainers. If that option is not
available, contact the maintainer through the GitHub profile without posting
technical details publicly.

Please include only the information needed to reproduce and assess the issue:

- affected extension and Forge Neo versions;
- operating system, GPU, and relevant runtime configuration;
- a minimal reproduction;
- expected and observed behavior;
- potential impact;
- suggested mitigation, if known.

Reports are reviewed on a best-effort basis. Confirmed issues will be handled
privately until a fix or safe disclosure plan is available.

## Scope

Security reports may cover the extension's local file handling, prompt/persona
storage, Forge integration, model-resource lifecycle, or unintended network or
data exposure. General generation quality, model behavior, and ordinary feature
requests belong in the public issue tracker.
