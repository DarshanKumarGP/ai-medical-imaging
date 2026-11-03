"""Remove studies that were skipped or failed earlier (and old CT rows) so the service analyses them again.
Only studies still waiting for review are removed, so no audit entries are affected."""
from db import connect

conn = connect()
cur = conn.execute(
    "DELETE FROM studies WHERE review_status='pending' AND (modality='CT' OR status='error')")
conn.commit()
print("Removed", cur.rowcount, "study row(s). The inference service will analyse them again.")