import SpaceGame as game
from DQN import qnn
from torchsummary import summary
import torch
import pygame

MODEL_PATH = 'best_Q_fn.pth'

Q = qnn().to('cuda')  # move model to GPU
GAME = game.SpaceInvaderGame()

def load_model(model, path=MODEL_PATH):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

def sample_actions(state):
    state = torch.tensor(state, dtype=torch.float32).to('cuda')
    return torch.argmax(Q(state)).item()

def play():
    state = GAME.reset()
    done = False
    score = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        action = sample_actions(state)
        next_state, reward, done, score = GAME.step(action)
    return score

if __name__ == '__main__':
    load_model(Q, MODEL_PATH)
    Q.eval()
    print(play())
    #summary(Q, input_size=(11,), device='cuda')
