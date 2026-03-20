# Forensic Report Template

## Status: To be defined with user

## Planned Sections
1. Cover page: Report ID, date, image hash, analyst info
2. Executive Summary: Final verdict, confidence, one-paragraph summary
3. Agent Score Table: Per-agent anomaly scores in a formatted table
4. Visual Evidence: Grad-CAM heatmap, face crop, landmark overlay
5. Metadata Analysis: EXIF data, ELA (Error Level Analysis) results
6. Methodology: Brief description of each agent and detection method
7. Appendix: Raw JSON output from all agents

## Report ID Format
`DFA-{YYYY}-TC-{random_6_char_hex}`

## Implementation
Use ReportLab for PDF generation.
Template to be built once report structure is agreed upon.
