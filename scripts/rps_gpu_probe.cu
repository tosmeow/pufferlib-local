#define ENV_HEADER "ocean/rock_paper_scissors/rock_paper_scissors.h"
#define PUFFER_ENV_NAME "rock_paper_scissors"
#include "../src/pufferl.cu"

int main(int argc, char** argv) {
    assert(argc == 3 || argc == 5);
    Ini ini = {};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, 0, NULL);
    if (argc == 5) {
        puf_ini_put(&ini, "policy.hidden_size", argv[3]);
        puf_ini_put(&ini, "policy.num_layers", argv[4]);
    }
    int hidden = (int)puf_ini_get(&ini, "policy", "hidden_size");
    int layers = (int)puf_ini_get(&ini, "policy", "num_layers");
    puf_ini_put(&ini, "vec.total_agents", "10");
    puf_ini_put(&ini, "vec.num_buffers", "1");
    puf_ini_put(&ini, "vec.num_threads", "1");
    puf_ini_put(&ini, "vec.num_policies", "1");
    puf_ini_put(&ini, "vec.hist_policy_percent", "0");
    puf_ini_put(&ini, "train.minibatch_size", "64");
    puf_ini_put(&ini, "base.async", "0");
    puf_ini_put(&ini, "base.cudagraphs", "-1");
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};
    PuffeRL* p = create_pufferl(&ini, &ctx);
    pufferl_load_policy(p, 0, argv[1]);
    Policy* pol = &p->policies[0];
    Prec input = {.shape = {10, 6}};
    assert(cudaMalloc((void**)&input.data, 60 * sizeof(float)) == cudaSuccess);
    assert(cudaMemset(pol->buffer_states[0].data, 0, layers*10*hidden*sizeof(float)) == cudaSuccess);
    FILE* out = fopen(argv[2], "w");
    assert(out);
    fprintf(out, "agent,step,p_rock,p_paper,p_scissors\n");
    for (int t = 0; t < 1000; t++) {
        float obs[60] = {0}, logits[40];
        for (int b = 0; b < 10; b++) {
            int code = t == 0 ? b - 1 : (t * 7 + b * 5 + t / 7) % 9;
            if (code >= 0) {
                obs[b*6 + code/3] = 1;
                obs[b*6 + 3 + code%3] = 1;
            }
        }
        assert(cudaMemcpy(input.data, obs, sizeof(obs), cudaMemcpyHostToDevice) == cudaSuccess);
        Prec dec = arch_forward(&pol->arch, pol->weights, pol->buf_acts[0],
            input, pol->buffer_states[0], p->streams[0]);
        assert(cudaStreamSynchronize(p->streams[0]) == cudaSuccess);
        assert(cudaMemcpy(logits, dec.data, sizeof(logits), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int b = 0; b < 10; b++) {
            double q[3], sum = 0;
            float mx = fmaxf(logits[b*4], fmaxf(logits[b*4+1], logits[b*4+2]));
            for (int a = 0; a < 3; a++) { q[a] = exp((double)logits[b*4+a]-mx); sum += q[a]; }
            fprintf(out, "%d,%d,%.10g,%.10g,%.10g\n", b, t, q[0]/sum, q[1]/sum, q[2]/sum);
        }
    }
    fclose(out);
    cudaFree(input.data);
    close_pufferl(p);
}
