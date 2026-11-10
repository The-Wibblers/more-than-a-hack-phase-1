"""SkillCorner's open games, normalised into one bundle per game.

    python -m external.download skillcorner      # fetch the raw files into data/external/raw/
    python -m external.build skillcorner         # write data/external/games/<game>/

Every game becomes the same four files, so the viewer needs one reader:

    meta.json      teams, periods, objects (index 0 is the ball), what the game has, licence
    tracking.f32   float32 [n_frames, n_objects, 3]: metres from the top-left corner, y down
                   (the viewer's own pitch convention), z only for the ball; NaN where unseen
    state.u8       uint8 [n_frames]: ball alive / owning team, where the source says
    events.json    events on the same frame index, coordinates in the same metres

See catalogue.py for where it is hosted and its licence.
"""
