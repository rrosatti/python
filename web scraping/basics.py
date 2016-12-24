import bs4 as bs
import urllib.request

sauce = urllib.request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()

soup = bs.BeautifulSoup(sauce, 'lxml')

#print(soup.title)
#print(soup.title.string)

# print the first paragraph
#print(soup.p)

# print all the paragraph tags
#print(soup.find_all('p'))

for p in soup.find_all('p'):
	print(p.text)
	#print(p.string)

#print(soup.get_text())

for url in soup.find_all('a'):
	#print(url)
	print(url.get('href'))