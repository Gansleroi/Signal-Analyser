from cli import Cli

if __name__ == "__main__":

    while True:
        Cli()

        ans = input("Начать заново?[y / n]\n")
        if ans.strip().lower() == 'n':
            print("До свидания!")
            break
