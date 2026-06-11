# Security Policy

## Reporting a vulnerability

Please do not disclose security vulnerabilities in a public issue. Report them privately through GitHub's security advisory feature for this repository, including reproduction steps and the affected version or commit.

ASTRID processes user-supplied datasets locally. Reports should pay particular attention to archive handling, parser behavior, generated HTML, dependency vulnerabilities, and accidental disclosure of dataset contents.

## Operational guidance

- Run ASTRID in an isolated environment when auditing untrusted datasets.
- Keep dependencies current and review automated dependency alerts.
- Do not publish generated audit reports until they have been checked for sensitive information.
- Treat heuristic findings as supporting evidence, not legal or compliance certification.
