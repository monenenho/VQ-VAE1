# ここではシンプルなprintベースのロガーを用意します
# wandb等を使う場合はここを拡張してください
def log(message: str):
    print(message)
