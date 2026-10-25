# Ethics, privacy and responsible use

Research/educational prototype only. Not intended for diagnosis, medical decisions or pediatric clinical monitoring. It does not predict asthma exacerbations.

Use real recordings only with appropriate permission and data-use authorization. Human-participant collection/evaluation may require institutional ethics review; this code and any training course do not authorize a study. Never scrape patient footage or download restricted datasets for this demonstration. No real-person data are included.

All processing is local. Streamlit binds to localhost, requires no account and disables usage telemetry. Upload copies are stored under a temporary directory and deleted after analysis, including expected failure paths. Upload bytes can remain in session memory until the uploader/session is cleared. Derived signals/results remain in session state until cleared. Reports may contain sensitive information and are not guaranteed anonymous.

The manifest adapter uses subject IDs internally for grouping; public outputs contain only sequential recording/condition codes, metrics and diagnostics. It omits original recording IDs, subject IDs, paths, source descriptions and condition free text. Do not regard pseudonymization or name removal as anonymization. The original frozen baseline retains path metadata for compatibility; do not publish its raw output for real recordings.

Keep identifiable footage, manifests, credentials and real-data outputs outside Git, with appropriate access control, retention and deletion rules. `.gitignore` excludes private/real directories, uploads, secrets, virtual environments, bytecode and checkpoints. Do not publish this local server to the network as a clinical service. No cloud deployment is part of this project.
