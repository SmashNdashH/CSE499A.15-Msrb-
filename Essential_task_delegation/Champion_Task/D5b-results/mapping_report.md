# D5b mapping report (xBD tier3)

2,635 tier3 tiles that DisasterM3 already uses: xBD polygons drawn by this notebook against DisasterM3's own masks. Each row shows where one xBD label's pixels go in DisasterM3.

```
    xbd_label   pixels  to_background  to_intact  to_damaged  to_destroyed      maps_to  consistency
    no-damage 63004843         0.0274     0.9726      0.0000        0.0000       Intact       0.9999
 minor-damage  5028159         0.0291     0.0008      0.9701        0.0000      Damaged       0.9991
 major-damage  3417416         0.0300     0.0003      0.9697        0.0000      Damaged       0.9997
    destroyed  3754594         0.0305     0.0001      0.0001        0.9692    Destroyed       0.9997
un-classified  1593004         0.9995     0.0004      0.0001        0.0000 255 (ignore)       0.8026
```

Pixel agreement after mapping: 97.2%. Median building IoU per tile: 0.905.

Lookup table (xBD code 0-5 -> class, 255 = ignore): `[0, 1, 2, 2, 3, 255]`, the same as D5's.