import re
comando = input("Entre com a expressão regular: ")
arquivo = input("Qual é o arquivo? ")
text = open(arquivo)
cont = 0
for l in text:
    l = l.rstrip()
    if re.search(comando, l):
        cont = cont + 1
print(cont)
