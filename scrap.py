from lxml import html
from lxml.etree import tostring
from itertools import chain

import requests

base_url = 'http://www.oldielyrics.com'
paul_url = 'http://www.oldielyrics.com/p/paul_mccartney_wings.html'
john_url = 'http://www.oldielyrics.com/j/john_lennon.html'




def scrap(url, csv) :
    page = requests.get(url)

    #print(page.content)

    tree = html.fromstring(page.content)

    songs = tree.xpath('//ol/li/a')

    print('title; text; author')


    for song in songs :

        page1 = requests.get(base_url+song.attrib['href'].replace('..', ''))
        content = page1.content
        content = content.replace(b"<br />", b"||")
        content = content.replace(b"<p>", b"")
        content = content.replace(b'\n', b'')
        content = content.replace(b'\r', b'')
        content = content.replace(b'  ', b'')

        #page1.content.replace(b"<br />", b"||").replace(b"<p>", b"").replace(b"</p>", b'').replace(b'\n', b'').replace(b'\r', b'')

        lyr =  html.fromstring(content)
        text = lyr.xpath('//div[@class="lyrics"]')
        #writer = lyr.xpath('//div[@class="meta_l"]')

        #print(page1.content)

        if len(text) > 0 and song is not None and text[0].text is not None:
            print(song.text + '; ' + text[0].text + '; 1')

    #print songs

def main():

    scrap(john_url, 1)

    return


if __name__ == '__main__':
    main()