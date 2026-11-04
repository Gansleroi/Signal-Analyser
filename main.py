from cli import Cli
from cli import get_choice

if __name__ == "__main__":

    while True:
        Cli()

        ans = get_choice("Начать заново?[y / n]\n", ['y', 'n'])
        if ans == 'n':
            print("До свидания!")
            break
