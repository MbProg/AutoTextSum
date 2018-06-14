The hMDS Corpus
The hMDS corpus[1] is a heterogeneous multi-document summarization corpus built with a novel corpus construction approach. This folder contains the hMDS corpus in version 2017-01-09. This version consists of 91 topics coming from 3 different domains. Further information about the corpus can be found at the corpus github page at https://github.com/AIPHES/hMDS.

Folder Structure
The folder is organized as follows: The root folder contains subfolders according to the used language in the topics, e.g. "english" (topics in German will be added soon). The folders in the language-subfolders contain the documents belonging to one particular topic. The file "topics.txt" contains a list with all topics for the particular language. Each topic belongs to one domain. Folder "D02T13", for example, contains the files for Topic 13, which belongs to Domain 02. The 3 domains are derived from 4 Wikipedia featured article overview pages: 

Domain 1: https://en.wikipedia.org/wiki/Wikipedia:Featured_articles#Art.2C_architecture.2C_and_archaeology
Domain 2: https://en.wikipedia.org/wiki/Wikipedia:Featured_articles#History
Domain 3: https://en.wikipedia.org/wiki/Wikipedia:Featured_articles#Law + https://en.wikipedia.org/wiki/Wikipedia:Featured_articles#Politics_and_government

Each topic consists of 3 subfolders "input", "reference", and "metadata":

Input:
The "input" folder contains the input for the summarization system in different versions. The input files are aligned against the extracted information nuggets. The files "4.html", "M4.raw.txt", "M4.txt", "A4.raw.txt", "A4.txt", "V4.raw.txt", and "V4.txt" in the folder "D02T13", for example, contain the source text which belong to nugget #4 in topic "D02T13". The "*.html" files contain the unprocessed HTML file exactly in the form it was downloaded including HTML tags, javascript, comments, etc. The "M*.raw.txt" files contain the relevant contents of the HTML page which were manually extracted from the corresponding "*.html" files. The corresponding "M*.txt" file contains the very same text with additional sentence segmentations. Each line in the "M*.txt" files contain exactly one sentence. "A*.raw.txt" and "A*.txt" files are similar to the M*-files except that the extraction of the relevant content was not extracted manually but automatically with the software Boilerpipe[2]. The files "V*.raw.txt" and "V*.txt" contain all visible text of the HTML page and are therefore the most verbose version of the corpus.

Reference:
The "reference" folder contains the reference summary in 2 different versions: The file "summary.raw.txt" contains the unprocessed reference summary and the file "summary.txt" contains the same text with additional sentence splitting. Each line in "summary.txt" contains exactly one sentence. Since the sentence splitting was performed automatically, wrong segmentations may be contained in the file.

Metadata:
The "metadata" folder contains metadata for the topic. For each topic, the "nugget.txt" file contains all the nuggets for a particular topic. Each line contains one nugget. The format for each line is <nugget ID>:<nugget text>. Each nugget metadata file (e.g. "4.txt") has 5 lines: Line 1 has the format <nugget ID>:<source document URL>, where <source document URL> equals the URL of the source document stored in Web Archive (https://archive.org/web). The second row contains the text in the source file which matches the nugget text contained in the summary. The third row contains the id of the document category as well as its name. The categories are further described in the annotation guidelines document (https://github.com/AIPHES/hMDS/blob/master/Guidelines.md). The fourth row contains the title of the page and the fifth row contains information about the license of the page. It contains "unknown" if no information about the license could be found on the page.

Example folder structure:
hMDS
 - english
   - topics.txt
   - ...
   - D02T12
   - D02T13
     - input
	   - ...
	   - 4.html
	   - ...
	   - M4.raw.txt
	   - M4.txt
	   - ...
	   - A4.raw.txt
	   - A4.txt
	   - ...
	   - A4.raw.txt
	   - A4.txt
	   - ...
	 - metadata
	   -
	 - reference
	   - summary.raw.txt
	   - summary.txt
   - D02T14
   - ...
 - german

References:
[1] Markus Zopf, Maxime Peyrard, and Judith Eckle-Kohler. 2016. The Next Step for Multi-Document Summarization: A Heterogeneous Multi-Genre Corpus Built with a Novel Construction Approach. In Proceedings of the 26th International Conference on Computational Linguistics (COLING 2016), pages 1535-1545, Osaka, Japan.
[2] Christian Kohlschütter, Peter Fankhauser, and Wolfgang Nejdl. 2010. Boilerplate detection using shallow text features. In Proceedings of the Third ACM International Conference on Web Search and Data Mining, pages 441–450, New York City, NY USA.