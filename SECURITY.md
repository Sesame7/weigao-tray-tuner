# Security Policy

## Reporting a Vulnerability

If you discover a security issue, please report it privately by email:

- `sesame.lu@alu.uestc.edu.cn`

Anonymous reports are accepted. If possible, include contact details so we can
follow up and coordinate fixes.

Please include:

- Clear issue description and impact
- Reproduction steps
- Affected files/modules
- Relevant logs, screenshots, or proof-of-concept

## Response SLA

We aim to:

- Acknowledge receipt within 48 hours
- Provide an initial assessment within 7 days

## Coordinated Disclosure

Please allow reasonable time for investigation and remediation before public
disclosure. We will coordinate timelines and credit where appropriate.

## Scope

In scope:

- App entry and controller (`main.py`, `app/controller.py`)
- UI modules (`ui/*`)
- Core utility modules (`core/*`)
- Parameter loading/saving and local config file handling
- Bundled/synced detector code as distributed in this repository (`detect/*`)

Out of scope:

- Insecure local deployment or machine-level environment issues
- Third-party dependency vulnerabilities (report upstream where appropriate)
- Vulnerabilities that exist only in `VisionRuntime` but are not present in this repository state


