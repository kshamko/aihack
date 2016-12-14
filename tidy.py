import csv

crap_words = [
    'oh no oh no', 'yeah yeah', 'ya ya', 'uh huh', 'la la',
    'da da', 'oh yeah oh yeah', 'oh no no', 'oo yeah', 'ooh ooh',
    'ah ha', 'mmm', 'mm mm', 'wo wo', 'oh oh'
]

def clean_text(text):
    x = text.replace("||", ' ')
    x = x.replace(',', '')
    x = x.replace('.', '')
    x = x.replace('-', '')
    x = x.replace(',', '')
    x = x.replace('"', '')
    x = x.replace('_', '')
    x = x.replace('?', '')
    x = x.replace('!', '')
    return x.lower()

def crap_count(text):
    crap = 0

    for w in crap_words:
        if w in text:
            crap += 1

    return crap

def main():

    csv_file = 'data/dataset.csv'

    with open(csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")

        tmp = []
        i = 0
        for row in sorted(reader):
            tmp.append(row)
            i = i + 1
        #    print row['title'] + '; ' + row['text'] + ';' #+ str(row['author'])

        print 'title; text; crap; lines; author'
        for r in sorted(tmp):
            text =  clean_text(r[1])
            crap = crap_count(text)
            print r[0] + '; ' + text + '; ' + str(crap) + '; ' + str(r[1].count('||') - 1) + '; ' + r[2]
    return


if __name__ == '__main__':
    main()