#pragma once

#include <ATen/Context.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sdp {

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug);
SDPBackend select_sdp_backend(sdp_params const& kernel_params);

} // namespace sdp
