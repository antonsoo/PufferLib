// cc -O2 -I. -Isrc -Iraylib-5.5_linux_amd64/include tests/test_hex_signals.c \
//   raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -o /tmp/test_hex_signals
#include <assert.h>
#include "ocean/hex/hex.h"

void check_next_game(Hex* env) {
    assert(env->tick == 0);
    for (int i = 0; i < TOTAL_CELLS; i++) {
        assert(env->board[i] == 0);
    }
    env->agents[0].actions[0] = 0;
    puf_step(env);
    assert(env->tick == 1);
    assert(env->board[0] == PLAYER_COLOR);
    assert(env->agents[0].rewards[0] == 0);
    assert(env->agents[0].terminals[0] == 0);
}

int main(void) {
    float obs[OBS_SIZE] = {0}, action = 0, reward = 0, terminal = 0;
    Hex env = {0};
    env.agents[0] = (Agent){
        .observations = obs, .actions = &action,
        .rewards = &reward, .terminals = &terminal,
    };
    env.rng = 42;
    puf_reset(&env);

    // CPU evaluation calls puf_step directly, without clearing these buffers.
    action = -1;
    puf_step(&env);
    assert(reward == -1 && terminal == 1);
    assert(env.log.n == 1);
    check_next_game(&env);

    // A player win also preserves its reward for exactly one transition.
    puf_reset(&env);
    for (int r = 0; r < BOARD_SIZE - 1; r++) {
        assert(!place_stone_and_check_win(&env, r * BOARD_SIZE, PLAYER_COLOR));
        assert(!place_stone_and_check_win(&env, r * BOARD_SIZE + 5, ENV_COLOR));
    }
    env.tick = BOARD_SIZE - 1;
    action = (BOARD_SIZE - 1) * BOARD_SIZE;
    puf_step(&env);
    assert(reward == 1 && terminal == 1);
    assert(env.log.n == 2);
    check_next_game(&env);
    puts("Hex terminal rewards last exactly one transition");
}
