# Queued jobs before the VPN switch

laneA | galvani | smoke 3136550 -> array 3136559 (32 tasks, %16) -> on-cluster collector 3136560; results in /mnt/lustre/work/gehler/gfh527/ni26_lane_a_returning_20261119/summary.json | ~22:00 CET (queue-dependent; smoke gates the array, unconfirmed while pending) | ready for VPN-off: yes (risk: smoke still PENDING; if it fails nothing runs and cannot be fixed offline)
NI-V | galvani | v5.1b 3136396+3136528, cap250 3136444, cap250-seed1 null 3136445, returning arm cap250 ret 3136529/3136545 fa 3136530/3136546, v5.1b ret 3136561 fa 3136562, parity 3136516, on-cluster collectors 3136563 (report) 3136580 (parity), release ni26_niv_v1_20261119/out/report_cap250_vs_v51b.json | ~18:00 CET (2080 queue is busy; ~145 short tasks) | ready for VPN-off: yes
