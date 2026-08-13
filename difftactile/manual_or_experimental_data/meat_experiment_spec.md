each of the 10 steaks is 5 mm thick

This is the record of what was **recorded**: all 23 runs, including repeats of the
same condition at different sensor heights. Only **10** of them are used by the
model (the union of `MEAT_TRAIN_TRIALS` and `MEAT_VALIDATION_TRIALS` in
`difftactile/cnn/dataset.py`) and only those 10 ship in the Zenodo bundle; the
other 13 were dropped rather than shipped for an end user to puzzle over. They
are marked **[shipped]** below.

On disk the shipped trials carry a descriptive prefix — `20260228-232013` is the
directory `1-metal-straw-beneath-1-steak-20260228-232013` — so the condition is
readable without consulting this file. The timestamp remains the trial's
identity; the prefix is only a label. Note the descriptions here say "straw" for
what the directory names call "metal straw", to distinguish them from the one
silicone-straw trial.

- 20260228-230937: straw on top (z=32) **[shipped]**
- 20260228-231646: straw beneath 1 steak (z=32)
- 20260228-231939: straw beneath 1 steak (z=31)
- 20260228-232013: straw beneath 1 steak (z=30) **[shipped]**
- 20260228-232534: straw beneath 2 steaks (z=30)
- 20260228-232632: straw beneath 2 steaks (z=32) **[shipped]**
- 20260228-233031: straw beneath 3 steaks (z=30) **[shipped]**
- 20260228-233454: straw beneath 4 steaks (z=30) **[shipped]**
- 20260228-233825: straw beneath 6 steaks (z=30)
- 20260228-233958: straw beneath 6 steaks (z=29)
- 20260228-234106: straw beneath 6 steaks (z=28)
- 20260228-234219: straw beneath 6 steaks (z=27)
- 20260228-234337: straw beneath 6 steaks (z=26) **[shipped]**
- 20260228-234613: no straw (z=26)
- 20260228-234824: no straw (z=25) **[shipped]**
- 20260228-235519: 2 straws beneath 2 steaks (z=32)
- 20260228-235639: 2 straws beneath 2 steaks (z=30)
- 20260228-235749: 2 straws beneath 2 steaks (z=29) **[shipped]**
- 20260301-000249: 3 straws beneath 2 steaks (z=32) - noisy signal, only 1 straw visible
- 20260301-000537: 3 straws beneath 2 steaks (z=30) - only 2 straws visible
- 20260301-000735: 3 straws beneath 2 steaks (z=28) - only 2.5 straws visible
- 20260301-000849: 3 straws beneath 2 steaks (z=26) - 3 straws visible **[shipped]**
- 20260301-001457: 1 silicone straw beneath 2 steaks (z=26) **[shipped]**
