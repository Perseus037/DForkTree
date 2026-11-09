# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
_registered = False

class TreeSpec(CustomSpecAlgo):

    def handle_server_args(self, server_args):
        SpeculativeAlgorithm.DFLASH.handle_server_args(server_args)
        from sglang.srt.arg_groups.overrides import declare_resolution
        declare_resolution(server_args, 'fc1_chain_tree', speculative_eagle_topk=2)
        if __import__('os').environ.get('FC1_TREE64') == '1':
            declare_resolution(server_args, 'fc1_tree64', speculative_num_draft_tokens=int(__import__('os').environ.get('FC1_TREE_N', '64')), speculative_num_steps=int(__import__('os').environ.get('FC1_DRAFT_BLOCK', '24')) - 1)
for name, value in vars(SpeculativeAlgorithm).items():
    if name.startswith(('is_', 'supports_')) or name in ('need_topk', 'has_draft_kv', 'carries_draft_hidden_states', 'create_future_map', 'build_disagg_draft_input', 'get_num_tokens_per_req_for_target_verify'):

        def delegate(self, *args, _name=name, **kwargs):
            return getattr(SpeculativeAlgorithm.DFLASH, _name)(*args, **kwargs)
        setattr(TreeSpec, name, delegate)

def register():
    global _registered
    if _registered:
        return
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    if __import__('os').environ.get('FC1_FUSED') == '1':
        from fc1_fused import install
        install()
        from fc1_conv import install as install_conv
        install_conv()
        from fc1_plan import install as install_plan
        install_plan()
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    for attr in ('retrieve_next_token', 'retrieve_next_sibling', 'retrieve_parent_token'):
        if not hasattr(DFlashVerifyInput, attr):
            setattr(DFlashVerifyInput, attr, None)
    if __import__('os').environ.get('FC1_BRANCH_PILOT') == '1':
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        original = DecodeCudaGraphRunner.get_spec_info

        def tree_capture_info(self, num_tokens):
            if self.model_runner.is_draft_worker:
                return original(self, num_tokens)
            return EagleVerifyInput(draft_token=None, custom_mask=self.buffers.custom_mask, positions=None, retrieve_index=None, retrieve_next_token=None, retrieve_next_sibling=None, retrieve_cum_len=None, spec_steps=int(__import__('os').environ.get('FC1_DRAFT_BLOCK', '24')) - 1, topk=2, draft_token_num=self.speculative_num_draft_tokens, capture_hidden_mode=CaptureHiddenMode.FULL, seq_lens_sum=None, seq_lens_cpu=None)
        DecodeCudaGraphRunner.get_spec_info = tree_capture_info

    @SpeculativeAlgorithm.register('FC1_DFLASH_TREE', supports_overlap=True, spec_class=TreeSpec)
    def factory(server_args):
        from fc1_dflash_worker import DFlashTreeWorkerV2
        return DFlashTreeWorkerV2
    _registered = True

def _verify_width(self, num_draft_tokens, is_draft_worker):
    if __import__('os').environ.get('FC1_TREE64') == '1' and is_draft_worker:
        return int(__import__('os').environ.get('FC1_DRAFT_BLOCK', '24'))
    return num_draft_tokens
TreeSpec.get_num_tokens_per_req_for_target_verify = _verify_width
