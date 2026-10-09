// Exercise the shipped PPO kernel, including roundoff inside the clip interval.
// Build like src/pufferl.cu with PRECISION_FLOAT, without PUFFERLIB_BUILD_MAIN.
#include "src/pufferl.cu"

struct ValueCase {
    float old_value, prediction, target, derivative;
};

int main() {
    ValueCase cases[] = {
        // old_value + (prediction - old_value) can round away from prediction.
        {0.2f, 0.02f, 0.5f, -0.48f},
        {-0.2f, -0.02f, -0.5f, 0.48f},
        {0.0f, 0.125f, 1.0f, -0.875f},
        // Outside the interval, the clipped branch can win or lose.
        {0.0f, 1.0f, 2.0f, 0.0f},
        {0.0f, -1.0f, -2.0f, 0.0f},
        {0.0f, 1.0f, -0.5f, 1.5f},
        {0.0f, -1.0f, 0.5f, -1.5f},
        // Both interval endpoints retain the derivative of clamp.
        {0.0f, 0.25f, 1.0f, -0.75f},
        {0.0f, -0.25f, -1.0f, 0.75f},
    };
    int n = sizeof(cases) / sizeof(cases[0]);
    // Unified memory keeps the test focused on the loss, not upload plumbing.
    float* data;
    assert(cudaMallocManaged(&data, (8 * n + LOSS_N) * sizeof(float)) == cudaSuccess);
    memset(data, 0, (8 * n + LOSS_N) * sizeof(float));
    float* logits = data;
    float* values = logits + 2 * n;
    float* returns = values + n;
    float* zeros = returns + n;
    float* grad_values = zeros + n;
    float* grad_logits = grad_values + n;
    float* mask = grad_logits + n;
    float* partials = mask + n;
    int* sizes;
    assert(cudaMallocManaged(&sizes, sizeof(int)) == cudaSuccess);
    sizes[0] = 1;
    for (int i = 0; i < n; i++) {
        logits[2 * i + 1] = cases[i].prediction;
        values[i] = cases[i].old_value;
        returns[i] = cases[i].target;
        mask[i] = 1.0f;
    }
    PPOKernelArgs args = {
        .grad_logits = grad_logits, .grad_values_pred = grad_values,
        .logits = logits, .values_pred = logits + 1,
        .act_sizes = sizes, .action_mask = mask, .num_atns = 1,
        .clip_coef = 0.2f, .vf_clip_coef = 0.25f, .vf_coef = 0.5f,
        .ent_coef = zeros, .T_seq = n, .A_total = 1, .N = 1,
        .is_continuous = false,
    };
    PPOGraphArgs graph = {
        .actions = zeros, .old_logprobs = zeros, .advantages = zeros,
        .values = values, .returns = returns,
    };
    ppo_loss_compute<<<1, PPO_THREADS>>>(partials, args, graph);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = 0.5f * cases[i].derivative / n;
        if (fabsf(grad_values[i] - expected) > 1e-7f) {
            fprintf(stderr, "case %d: value gradient %.9g, expected %.9g\n",
                i, grad_values[i], expected);
            failures++;
        }
    }
    cudaFree(sizes);
    cudaFree(data);
    if (failures) {
        return 1;
    }
    printf("PPO value gradients: %d cases passed\n", n);
    return 0;
}
