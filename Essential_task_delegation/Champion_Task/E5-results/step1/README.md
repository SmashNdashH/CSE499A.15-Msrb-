# E5 step 1: the validation leak and the benchmark's copies of training images

20,988 DisasterM3 Instruct files and 7,278 benchmark files were fingerprinted in 15.4 min.

## The validation leak (the 645 pairs every D4 run is scored on)

- **Leaked pairs (counted): 40**, holding **155,812** of 1,808,111 destroyed pixels (8.6%). The audit says 40 and 155,812; the PC check (xBD's original files) found 21 and 80,709.
  - same before-image as a training pair: 40; same after-image: 0
- The PC check's 21 among them: 21 of 21.
- Near-copies of before-images, not counted: 0 near copy, 3 review (pictures in `review/`).
- Twins that are both in validation (no leak): 4; twins outside the pool: 1.
- Shared before-images in the pool: 250 groups (the audit: 264).
- Leaked twins' after-images correlate at a median of 0.966 (the audit: 0.88).

## The benchmark's copies of Instruct images

- Benchmark files with a match in Instruct: 519 same pixels, 39 near copy, 48 review only.
- **Optical questions of the six tasks whose after-image is a copy: 186 of 10,675 (1.74%).**

| task   |   questions |   optical |   after_copied |   after_copied_optical |   before_only_copied |   before_only_copied_optical |   review_only |
|:-------|------------:|----------:|---------------:|-----------------------:|---------------------:|-----------------------------:|--------------:|
| DTR    |         420 |       353 |             10 |                     10 |                    6 |                            6 |             2 |
| BBR    |        2363 |      2006 |             35 |                     35 |                   24 |                           24 |            19 |
| BDC    |        4982 |      3583 |            188 |                     66 |                   44 |                           43 |            24 |
| DRE    |        2178 |      1868 |             34 |                     34 |                   23 |                           23 |            14 |
| DSR    |        2007 |      2007 |             35 |                     35 |                   24 |                           24 |            18 |
| ORR    |        1018 |       858 |              6 |                      6 |                    0 |                            0 |             6 |

By event:

| event                                   |   near copy |   review |   same pixels |
|:----------------------------------------|------------:|---------:|--------------:|
| hurricane_florence                      |           0 |        6 |             0 |
| la_palma_volcano                        |           0 |        0 |            95 |
| la_palma_volcano_00000011_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000020_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000029_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000039_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000047_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000056_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000065_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000074_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000083_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000091_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000100_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000109_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000118_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000127_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000135_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000144_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000153_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000162_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000171_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000179_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000188_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000197_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000206_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000214_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000223_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000232_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000241_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000250_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000258_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000267_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000276_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000285_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000294_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000302_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000315_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000324_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000333_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000347_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000355_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000364_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000383_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000391_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000400_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000409_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000419_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000428_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000445_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000454_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000463_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000472_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000480_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000489_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000507_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000516_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000524_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000534_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000543_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000552_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000561_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000570_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000579_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000593_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000602_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000610_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000620_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000629_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000638_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000647_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000656_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000665_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000676_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000696_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000704_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000714_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000723_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000734_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000742_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000751_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000761_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000772_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000781_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000790_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000800_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000810_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000820_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000829_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000838_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000849_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000860_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000869_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000878_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000886_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000896_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000907_pre_disaster  |           0 |        0 |             1 |
| la_palma_volcano_00000916_pre_disaster  |           0 |        0 |             1 |
| lower_puna_volcano                      |           0 |        1 |             0 |
| mexico_earthquake                       |           0 |        9 |             0 |
| mexico_earthquake_00000078_pre_disaster |           0 |        1 |             0 |
| mexico_earthquake_00000092_pre_disaster |           0 |        1 |             0 |
| mexico_earthquake_00000146_pre_disaster |           0 |        1 |             0 |
| moore_tornado                           |           0 |        1 |             0 |
| moore_tornado_00000193_pre_disaster     |           0 |        1 |             0 |
| myanmar_hurricane                       |           0 |        1 |             0 |
| myanmar_hurricane_00000025_pre_disaster |           1 |        0 |             0 |
| noto_earthquake                         |           0 |        1 |             0 |
| noto_earthquake_pre_                    |           0 |        1 |             0 |
| palu_tsunami                            |           0 |        0 |            56 |
| palu_tsunami_00000005_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000006_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000008_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000009_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000012_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000017_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000021_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000030_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000032_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000037_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000041_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000042_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000043_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000045_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000058_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000059_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000060_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000068_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000072_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000073_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000078_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000082_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000083_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000084_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000087_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000089_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000093_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000097_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000102_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000105_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000106_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000107_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000111_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000113_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000114_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000115_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000118_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000120_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000121_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000122_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000125_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000127_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000134_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000139_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000142_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000147_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000154_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000157_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000160_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000168_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000176_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000179_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000181_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000182_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000184_pre_disaster      |           0 |        0 |             1 |
| palu_tsunami_00000190_pre_disaster      |           0 |        0 |             1 |
| pinery_bushfire                         |           0 |        0 |             3 |
| portugal_wildfire                       |           0 |        1 |             0 |
| socal_fire                              |           5 |        5 |             0 |
| socal_fire_00000095_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000250_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000343_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000468_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000488_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000629_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000652_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000779_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000821_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00000942_pre_disaster        |           0 |        0 |             1 |
| socal_fire_00001135_pre_disaster        |           0 |        0 |             1 |
| spain_volcano                           |           0 |        0 |            48 |
| spain_volcano_00000031_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000032_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000062_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000063_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000093_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000094_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000124_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000125_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000155_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000156_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000186_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000187_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000217_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000218_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000248_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000249_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000279_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000280_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000311_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000312_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000374_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000404_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000405_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000435_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000466_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000467_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000497_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000528_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000529_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000560_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000622_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000623_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000653_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000654_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000684_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000715_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000716_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000746_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000747_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000777_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000778_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000808_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000809_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000839_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000840_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000870_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000871_pre_disaster     |           0 |        0 |             1 |
| spain_volcano_00000901_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami                           |           0 |        0 |            31 |
| sunda_tsunami_00000004_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000005_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000006_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000014_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000016_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000023_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000030_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000043_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000045_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000046_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000049_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000053_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000055_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000058_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000059_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000061_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000069_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000070_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000074_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000076_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000078_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000083_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000085_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000087_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000090_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000094_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000099_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000101_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000105_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000119_pre_disaster     |           0 |        0 |             1 |
| sunda_tsunami_00000146_pre_disaster     |           0 |        0 |             1 |
| woolsey_fire                            |          33 |       18 |             0 |
| woolsey_fire_00000000_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000024_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000033_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000052_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000084_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000093_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000121_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000127_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000136_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000146_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000167_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000206_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000235_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000246_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000262_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000271_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000317_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000328_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000409_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000428_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000448_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000464_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000473_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000480_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000522_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000541_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000555_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000568_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000585_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000598_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000622_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000635_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000684_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000699_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000728_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000740_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000749_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000763_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000780_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000791_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000803_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000817_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000833_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000846_pre_disaster      |           0 |        0 |             1 |
| woolsey_fire_00000863_pre_disaster      |           0 |        0 |             1 |

## Against the expectation (fixed before the run, 26 Oct)

- leaked pairs within 21-40: **yes**
- PC check's 21 all found: **yes**
- O2 needs a clean re-score (over 1% of optical questions with a copied after-image): **yes**

## How it was checked, and the alternatives

- **Same pixels** (the counted rule): an MD5 of the decoded pixels, so a copy saved again with other file bytes still matches. The file MD5 is kept too (`same_file_bytes`).
- **Near copy** (listed, not counted): dHash (64 bits from a 9 × 8 grey thumbnail) within 6 bits AND 32 × 32 thumbnails correlating at 0.95 or more. **Review**: within 10 bits and 0.85 or more. Images over half black or nearly flat are never near-matched, since dHash collides on them.
- **Not used:**
  - matching file names, which misses copies under another name; that is how D5's check missed this leak;
  - building masks: twins' masks differ (IoU 0.72), so they can't find copies;
  - local features (SIFT, ORB) or CNN embeddings, which survive bigger crops and shifts. Add them only if the pictures show copies these thresholds miss.
- **The split** is rebuilt with R11's verbatim cells and checked: 6,443 pairs, 645 validation images (560 xBD), 1,808,111 destroyed pixels.

## Files

- `val_train_leaks_hash.csv`: every validation pair with a match, and whether it's counted
- `pool_twin_groups.csv`: before-images shared by two or more pool pairs
- `bench_copies.csv`: every benchmark file with a match in Instruct
- `bench_questions_flagged.csv`: the six tasks' questions that use a copied image (IDs as in H1 V2 and L0)
- `fingerprints_instruct.csv`, `fingerprints_bench.csv`: every file's fingerprints, for later checks without a rerun
- `review/`: pictures, the benchmark or validation image on the left and its match in training on the right
- `summary.json`

## Next, on the PC

- Re-score R16c (steps 2 and 5) and, with E5 step 2's per-image counts, R11, R11s7, R9 and R1 without the counted pairs.
- If the benchmark share is over 1%: re-score the 7B's six tasks (L0, H1 V2) and H2 without the flagged questions.