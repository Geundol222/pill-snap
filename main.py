from matplotlib import rcParams

import src.train as train
import src.infer as infer

def main():
    rcParams['font.family'] = 'Malgun Gothic'
    rcParams['axes.unicode_minus'] = False

    train.train()
    infer.validation_score()
    infer.test_loop()

if __name__ == "__main__":
    main()