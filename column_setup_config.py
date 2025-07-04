# column_setup_config.py
# ------------------------------------------------------------------
# Hard-coded column-setup rules.  Extend or edit with Git as needed.
# ------------------------------------------------------------------
COLUMN_SETUP_CONFIG = [
    # ---- DEFAULT PROJECT ----------------------------------------------------
    # | Project | Target Column | Position | Reference Column | Default Value |
    {"Project": "Default", "Target Column": "Noise Tag",
     "Position": "before", "Reference Column": "Content", "Default Value": "3"},

    {"Project": "Default", "Target Column": "Official Account",
     "Position": "before", "Reference Column": "Author", "Default Value": "Non Official Account"},

    {"Project": "Default", "Target Column": "Issue",
     "Position": "after", "Reference Column": "Content", "Default Value": ""},

    {"Project": "Default", "Target Column": "Sub Issue",
     "Position": "after", "Reference Column": "Issue", "Default Value": ""},

    {"Project": "Default", "Target Column": "Followers",
     "Position": "after", "Reference Column": "Buzz", "Default Value": ""},

    {"Project": "Default", "Target Column": "Location",
     "Position": "after", "Reference Column": "Gender", "Default Value": ""},

    {"Project": "Default", "Target Column": "Media Tier",
     "Position": "before", "Reference Column": "Media Name", "Default Value": ""},

    {"Project": "Default", "Target Column": "Creator Type",
     "Position": "after", "Reference Column": "Author", "Default Value": ""},

    {"Project": "Default", "Target Column": "Rules Affected",
     "Position": "after", "Reference Column": "Location", "Default Value": ""},

    {"Project": "Default", "Target Column": "Rules Affected Words",
     "Position": "after", "Reference Column": "Rules Affected", "Default Value": ""},
]
