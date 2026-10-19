# D5 mapping report

1,792 overlapping tiles: xBD polygons drawn by this notebook against DisasterM3's own masks. Each row shows where one xBD label's pixels go in DisasterM3.

```
    xbd_label   pixels  to_background  to_intact  to_damaged  to_destroyed      maps_to  consistency
    no-damage 72900780         0.0212     0.9788      0.0000         0.000       Intact       1.0000
 minor-damage 14189745         0.0224     0.0001      0.9775         0.000      Damaged       0.9999
 major-damage 17281674         0.0205     0.0001      0.9794         0.000      Damaged       0.9999
    destroyed  6848976         0.0267     0.0001      0.0002         0.973    Destroyed       0.9997
un-classified  1096413         0.9996     0.0002      0.0001         0.000 255 (ignore)       0.6370
```

Pixel agreement after mapping: 97.8%. Median building IoU per tile: 0.929.

Lookup table (xBD code 0-5 -> class, 255 = ignore): `[0, 1, 2, 2, 3, 255]`