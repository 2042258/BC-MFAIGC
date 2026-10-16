# else:
#     # PoS
#     candidate_PoS_blocks = {}
#     print("select winning block based on PoS")
#     # filter the ordered_all_blocks_processing_queue to contain only the blocks within time limit
#     for (block_arrival_time, block_to_verify) in ordered_all_blocks_processing_queue:
#         if block_arrival_time < args['miner_pos_propagated_block_wait_time']:
#             candidate_PoS_blocks[devices_in_network.devices_set[block_to_verify.return_mined_by()].return_stake()] = block_to_verify
#     high_to_low_stake_ordered_blocks = sorted(candidate_PoS_blocks.items(), reverse=True)
#     # for PoS, requests every device in the network to add a valid block that has the most miner stake in the PoS candidate blocks list, which can be verified through chain
#     for (stake, PoS_candidate_block) in high_to_low_stake_ordered_blocks:
#         verified_block, verification_time = miner.verify_block(PoS_candidate_block, PoS_candidate_block.return_mined_by())
#         if verified_block:
#             block_mined_by = verified_block.return_mined_by()
#             if block_mined_by == miner.return_idx():
#                 print(f"Miner {miner.return_idx()} with stake {stake} is adding its own mined block.")
#             else:
#                 print(f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}.")
#             if miner.online_switcher():
#                 miner.add_block(verified_block)
#             else:
#                 print(f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain.")
#             if miner.return_the_added_block():
#                 # requesting devices in its associations to download this block
#                 miner.request_to_download(verified_block, block_arrival_time + verification_time)
#                 break