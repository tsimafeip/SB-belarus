# SB-Belarus-Yesterday
The main contribution of this project is a **new dataset** (data/sb_articles.db).<br>
I have scraped and preprocessed ~20k political articles, published by Belarusian state-owned newspaper "SB. Belarus' Segodnya" ("SB. Belarus Today").

As part of this project, I also completed other tasks:
- simple count-based analysis was performed to analyse this dataset (see code/helper.py)
- unsupervised LDA modelling for topic interpretation was conducted (see code/lda)
- detailed description of all steps and background overview of Belarusian political situation can be found [in the accompanying paper](meta/Framing and Topic Modelling in Belarusian State-Owned Media.pdf).

## Data Source and Examples
 Data for analysing was scraped from [web version of "SB. Belarus' Segodnya"](https://www.sb.by). 
 It is currently down, but you can find articles by hyperlink using [Internet Archive](http://web.archive.org).

 I use SQLite for storing data and querying it. 
 Data is stored in the single table, called 'documents', with the structure, described in the table above.
 I will illustrate each field with an example from actual data. 
 Please note, that source data is in Russian, I translated it by myself for this table.

| Field Name             | Description                                 | Example                                                                                                         |
|------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **document\_id**       | non-empty unique identifier of the document | *1440895*                                                                                                       |
| **title**              | primary document title                      | *To whom and why does union integration hinder*                                                                 |
| **title_h1**           | optional secondary document title           | *Our alternative*                                                                                               |
| **tags**               | optional list of tags selected by author    | *union state@politics@economy@sanctions*                                                                        |
| **similar\_documents** | optional links to similar articles          | *https://www.sb.by/articles/soyuz-belarus-rossiya*                                                              |
| **publication\_date**  | date and time of publication                | *2021-10-21T10:42:00+03:00*                                                                                     |
| **author**             | name of the author                          | *polina konoga*                                                                                                 |
| **body**               | main text of the article                    | *Alexei Avdonin: The West has chosen the Union State as its target.* (first sentence, full body is much longer) |
| **hyperlink**          | original url of the article                 | *https://www.sb.by/articles/nasha-alternativa-souz.html*                                                        |

 