import sys

if len(sys.argv) != 5:
    print("Usage: python script.py <benign_devices_folder> <malicious_devices_folder> <PoS_3_vh_008_run1_folder> <PoS_BCNN_folder>")
    sys.exit(1)

log_folder_b = sys.argv[1]
log_folder_m = sys.argv[2]
log_folder_PoS_3_vh_008_run1 = sys.argv[3]
log_folder_PoS_BCNN = sys.argv[4]