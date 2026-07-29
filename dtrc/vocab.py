"""MVP expert vocabularies (stage-2 teacher-forced training).

Full paper vocab can expand later; first milestone uses three experts only.
"""

# Correction experts (no dedicated "normal" expert — normal = zero gates).
VI_MVP_EXPERTS = ('low', 'over')
IR_MVP_EXPERTS = ('blur',)
