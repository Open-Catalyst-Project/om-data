#!/bin/bash

#SBATCH --job-name=mechdb_pipeline
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=learnaccel,scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-181

list=(1433 1447 1664 1666 1671 1707 1710 1716 1724 1729 1741 1745 1750 1762 1767 1780 1783 1787 1797 1800 1803 1822 1841 1842 1845 1847 1853 1883 1890 1897 1904 1906 1908 1909 1918 1919 1966 1972 1982 2008 2013 2014 2016 2020 2025 2030 2038 2052 2063 2095 2100 2129 2136 2140 2149 2151 2152 2159 2177 2180 2208 2211 2235 2257 2271 2295 2308 2310 2314 2321 2324 2326 2364 2369 2373 2387 2390 2394 2397 2401 2439 2485 25 2612 2779 2864 2950 3123 3126 3132 3138 3144 3150 3162 3174 3188 3204 3216 3222 3234 3240 3246 3254 3267 3285 3380 3396 3808 3847 3855 3875 3909 3943 3951 3971 3979 4003 4018 4030 4044 4049 4070 4090 4095 4130 414 4143 4155 4168 4180 4193 4213 4223 4256 4278 4286 4310 4325 4347 4373 4380 4393 4532 4538 4544 4550 4556 4562 4568 4574 4586 4592 466 4737 4766 4791 4799 4903 4911 4941 4949 5048 5062 5075 5088 5104 5106 5118 5122 5136 5162 5175 5293 5294 5383 5391 5399 5420 598 648 671 675)
idx=${list[SLURM_ARRAY_TASK_ID]}


echo /private/home/levineds/miniconda3/envs/mace/bin/python mechdb_pipeline.py --mechdb_sdfs_path /checkpoint/levineds/rmechdb --output_path /checkpoint/levineds/rmechdb/frames/ --start_index $idx --end_index $(($idx + 1))
rm -rf /checkpoint/levineds/rmechdb/frames/rmechdb_$idx
/private/home/levineds/miniconda3/envs/mace/bin/python mechdb_pipeline.py --mechdb_sdfs_path /checkpoint/levineds/rmechdb --output_path /checkpoint/levineds/rmechdb/frames/ --start_index $idx --end_index $(($idx + 1))
echo done
