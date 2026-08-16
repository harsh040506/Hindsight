"""
Hindsight -- closed-loop experimentation for automated content channels.

Automated channels publish hundreds of videos and learn nothing from any of
them. The uploads succeed, the views land somewhere between 40 and 2000, and
nobody can say why. Hindsight closes that loop:

    ingest    pull the published catalog from the YouTube Data API
    analyze   score every video against its own time-cohort, then measure
              which metadata choices actually moved the number
    report    render the findings as a standalone HTML page
    design    turn the biggest untested lever into a concrete A/B plan
    verdict   re-read the channel later and call the experiment

The unit of value is the experiment, not the dashboard. A dashboard tells you
what happened; Hindsight tells you what to change next and then holds itself
accountable for whether the change worked.

Hindsight is strictly read-only against YouTube. It requests no write scope
and calls no write endpoint -- it cannot upload, edit, delete, or publish
anything on your channel. See docs/ARCHITECTURE.md#safety.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
