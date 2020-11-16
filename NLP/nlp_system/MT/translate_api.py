from translate import Translator
import sys
translator= Translator(to_lang="zh")

def get_translation(input_str):
    translation = translator.translate(input_str)
    return translation

if __name__ == "__main__":
    argvs = sys.argv
    input_str = argvs[1]
    try:
        get_translation(input_str)
    except Exception as e:
        print(-1)