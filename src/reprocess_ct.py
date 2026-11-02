"""Remove old CT rows (made before the CT model existed) so the service analyses them again.
CT studies cannot be reviewed while skipped, so no audit entries are affected."""
from db import connect

conn = connect()
cur = conn.execute("DELETE FROM studies WHERE modality='CT' AND review_status='pending'")
conn.commit()
print("Removed", cur.rowcount, "CT row(s). Start the inference service to analyse them again.")