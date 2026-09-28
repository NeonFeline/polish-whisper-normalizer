"""Benchmark datasets representing various text workloads in Polish ASR.

Contains curated sets for:
- Short conversational utterances (voice assistant / command style)
- Medium turns (podcasts, news, conversational ASR)
- Long paragraphs (literary, article, lecture transcripts)
- Dense numeric sentences (finance, stats, accounting)
- Spoken time expressions (various Polish colloquial and formal time phrasing)
- Dates (all calendar months, full and partial dates)
- Diacritic-less texts (simulating diacritic-free ASR transcriptions)
- Plain literary Polish (no numbers or times, testing baseline regex passthrough)
- Stress and adversarial samples (punctuation storms, extreme lengths, special chars)
"""

from __future__ import annotations

SHORT_UTTERANCES: list[str] = [
    "włącz światło w kuchni",
    "która jest godzina",
    "obudź mnie o siódmej rano",
    "dodaj pięć jabłek do koszyka",
    "zadzwoń do mamy",
    "jaka będzie jutro pogoda",
    "ustaw minutnik na piętnaście minut",
    "wyłącz radio w salonie",
    "jeden dwa trzy próba mikrofonu",
    "zapłacę kartą zbliżeniowo",
    "gdzie jest najbliższa apteka",
    "poproszę dwa bilety normalne",
    "temperatura wynosi dwadzieścia stopni",
    "jest dwunasta w południe",
    "dojeżdżam za dziesięć minut",
    "spotkanie o szesnastej trzydzieści",
    "dwa plus dwa równa się cztery",
    "konto numer zero zero siedem",
    "kurs dolara wzrósł o dwa procent",
    "do widzenia i miłego dnia",
]

CONVERSATIONAL_MEDIUM: list[str] = [
    "Dzień dobry, chciałem zapytać o fakturę z piętnastego maja na kwotę stu dwudziestu pięciu złotych i pięćdziesięciu groszy.",
    "Pociąg do Krakowa odjeżdża z peronu trzeciego dokładnie o godzinie osiemnastej czterdzieści pięć, więc mamy jeszcze pół godziny.",
    "W zeszłym roku nasza firma zanotowała wzrost przychodów o dwadzieścia trzy procent, co daje ponad dwa miliony trzysta tysięcy euro.",
    "Profesor Nowak przyjmuje studentów we wtorki od dziesiątej do dwunastej w pokoju numer trzysta czternaście na drugim piętrze.",
    "Na skrzyżowaniu ulicy Marszałkowskiej i Świętokrzyskiej doszło do kolizji dwóch samochodów osobowych około ósmej rano.",
    "Wczoraj kupiłem półtora kilograma jabłek, dwa i pół litra mleka oraz trzy bochenki chleba za niespełna czterdzieści złotych.",
    "Konferencja odbędzie się dwudziestego drugiego października dwa tysiące dwudziestego piątego roku w auli głównej uniwersytetu.",
    "Zgodnie z prognozą ciśnienie spadnie do tysiąca dziesięciu hektopaskali, a wiatr osiągnie prędkość siedemdziesięciu kilometrów na godzinę.",
    "W zawodach wzięło udział stu pięćdziesięciu uczestników z osiemnastu krajów całego świata, w tym trzydziestu Polaków.",
    "Mieszkanie o powierzchni sześćdziesięciu czterech metrów kwadratowych kosztuje około siedmiuset pięćdziesięciu tysięcy złotych.",
    "Za dwadzieścia dwunasta zadzwonił kurier z informacją, że dostarczy paczkę między trzynastą a piętnastą.",
    "Według statystyk ponad osiemdziesiąt pięć procent Polaków korzysta z bankowości internetowej przynajmniej raz w tygodniu.",
]

LITERARY_LONG: list[str] = [
    (
        "W owym pamiętnym roku tysiąc osiemset dwunastym, gdy wielka armia ruszała na Moskwę, "
        "w małym dworku na Litwie przygotowywano się do hucznych uroczystości zaręczynowych. "
        "Gospodarz wstał o świcie, dokładnie o godzinie piątej rano, aby dopilnować wszystkich przygotowań. "
        "Zjechało się ponad stu dwudziestu gości z okolicznych zaścianków, w tym generałowie i oficerowie "
        "walczący pod dowództwem księcia Józefa Poniatowskiego. Uczta trwała nieprzerwanie przez trzy dni i noce, "
        "a na stoły wniesiono pieczone dziki, setki butelek wybornego wina oraz niezliczone półmiski tradycyjnych potraw. "
        "Koszt całego przedsięwzięcia oszacowano na blisko piętnaście tysięcy złotych monet, co w tamtych czasach "
        "stanowiło prawdziwą fortunę godną magnackiego rodu."
    ),
    (
        "Raport finansowy międzynarodowej korporacji technologicznej za trzeci kwartał bieżącego roku wykazał "
        "rekordowe zyski na poziomie czterech miliardów siedmiuset milionów dolarów. Stanowi to wzrost o osiemnaście "
        "i pół procenta w porównaniu z analogicznym okresem roku ubiegłego. Główne źródła przychodów stanowiły usługi chmurowe, "
        "które wygenerowały dwa miliardy trzysta milionów euro, oraz sprzedaż urządzeń mobilnych o wartości miliarda "
        "ośmiuset milionów dolarów. Jednocześnie wydatki na badania i rozwój wzrosły do dziewięciuset pięćdziesięciu "
        "milionów dolarów, co stanowi blisko dwadzieścia procent całkowitego budżetu operacyjnego spółki. Zarząd ogłosił "
        "wypłatę dywidendy w wysokości trzech złotych i siedemdziesięciu pięciu groszy na jedną akcję."
    ),
    (
        "Kolej transsyberyjska, licząca dokładnie dziewięć tysięcy dwieście osiemdziesiąt dziewięć kilometrów, "
        "jest najdłuższą linią kolejową na świecie, przecinającą osiem stref czasowych. Podróż ze stacji początkowej "
        "w Moskwie do Władywostoku nad Oceanem Spokojnym trwa zazwyczaj sześć dni, cztery godziny i dwadzieścia minut. "
        "Średnia prędkość pociągu wynosi około sześćdziesięciu pięciu kilometrów na godzinę, a po drodze skład zatrzymuje się "
        "na sześćdziesięciu czterech stacjach węzłowych. Każdego roku z tej trasy korzysta ponad trzysta pięćdziesiąt tysięcy "
        "pasażerów, przewożąc przy tym miliony ton towarów przemysłowych i surowców naturalnych."
    ),
]

NUMERIC_HEAVY: list[str] = [
    "sto dwadzieścia trzy tysiące czterysta pięćdziesiąt sześć",
    "dziewięćset dziewięćdziesiąt dziewięć milionów osiemset siedemdziesiąt sześć tysięcy pięćset czterdzieści trzy",
    "pięćset dwadzieścia trzy złote i czterdzieści osiem groszy",
    "tysiąc dwieście euro oraz siedemdziesiąt pięć centów",
    "czterdzieści pięć procent z sumy stu pięćdziesięciu tysięcy dolarów",
    "minus piętnaście stopni w nocy i plus dwadzieścia dwa w dzień",
    "trzy całe i czternaście setnych oraz dwa i pół miliona",
    "jedna trzecia, dwie piąte, trzy czwarte oraz pięć ósmych",
    "dwudziesty pierwszy maja roku tysiąc dziewięćset osiemdziesiątego czwartego",
    "o godzinie siedemnastej czterdzieści pięć zapłacono czterysta pięćdziesiąt złotych",
    "siedemdziesiąt osiem miliardów trzysta dwadzieścia milionów funtów",
    "dziesięć, sto, tysiąc, dziesięć tysięcy, sto tysięcy, milion",
]

TIME_HEAVY: list[str] = [
    "wpół do ósmej rano",
    "za piętnaście dwunasta w południe",
    "dziesięć po piątej po południu",
    "godzina dziewiętnasta trzydzieści pięć",
    "o dwudziestej trzeciej piętnaście",
    "za dwadzieścia minut czwarta",
    "kwadrans po szóstej wieczorem",
    "od ósmej rano do szesnastej trzydzieści",
    "spotkamy się o północy",
    "było dokładnie dziesięć po pierwszej w nocy",
    "o godzinie osiemnastej zero zero",
    "wpół do dwudziestej czwartej",
]

DATE_HEAVY: list[str] = [
    "pierwszego stycznia dwa tysiące dwudziestego czwartego roku",
    "trzeciego maja tysiąc siedemset dziewięćdziesiątego pierwszego roku",
    "piętnastego lipca tysiąc czterysta dziesiątego roku pod Grunwaldem",
    "jedenastego listopada tysiąc dziewięćset osiemnastego roku",
    "dwudziestego dziewiątego lutego roku przestępnego",
    "trzydziestego pierwszego grudnia o dwudziestej trzeciej pięćdziesiąt dziewięć",
    "piątego maja 2026",
    "3 maja 1791 r.",
    "21.05.2024",
    "01.09.1939 roku",
]

DIACRITICLESS: list[str] = [
    "trzysta czterdziesci osiem zlotych i piecdziesiat groszy",
    "dziewiecset dziewiecdziesiat dziewiec milionow",
    "wpol do osmej rano w srode",
    "za pietnascie dwunasta w poludnie",
    "piatego maja roku dwa tysiace dwudziestego szostego",
    "dwadziescia piec procent rabatu na wszystko",
    "bylismy tam o szesnastej trzydziesci piec",
    "jeden dwa trzy cztery piec szesc siedem",
]

PLAIN_POLISH_NO_NUM: list[str] = [
    "W szczebrzeszynie chrząszcz brzmi w trzcinie i szczebrzeszyn z tego słynie.",
    "Za górami za lasami mieszkał sobie stary leśnik z wiernym psem.",
    "Jesienny wiatr szeleścił suchymi liśćmi opadającymi z przydrożnych dębów.",
    "Przemierzając górskie szlaki podziwialiśmy malownicze zachody słońca.",
    "Książka leżała na dębowym stole obok zapalonej świecy i filiżanki herbaty.",
    "Spokojna rzeka leniwie płynęła przez zieloną dolinę otoczoną wzgórzami.",
    "Spotkaliśmy starych znajomych na rynku i rozmawialiśmy przez całe popołudnie.",
    "Nocne niebo rozbłysło milionami jasnych gwiazd nad cichą polaną.",
]

ADVERSARIAL_STRESS: list[str] = [
    # nested and unclosed brackets
    "test [nawias [zagnieżdżony]] oraz (okrągły (wielokrotny)) i <tag <inny>>",
    # repeated whitespace and periods
    "początek" + " " * 500 + "koniec",
    "kropki" + " . " * 100 + "po kropkach",
    "elipsa" + "..." * 50 + "po elipsie",
    # unusual unicode and mixed symbols
    "ceny: 100 zł / 25 € / 50 $ / 10 £ / 99 ¢ oraz 25.5% i -10 +20",
    "telefon: +48 123 456 789 oraz NIP: 123-456-78-90",
    "mieszane jednostki: 10kg, 20m, 5litrów, 15sekund, 100km/h",
    # long sequence of numbers
    " ".join(["jeden", "dwa", "trzy", "cztery", "pięć"] * 20),
    # punctuation storm
    "co??? naprawdę?!!?!?! ... no nie wiem... (chyba tak!)",
    # interjections
    "eee... no yyy... mhm... hmm... tak myślę... uh... um...",
]

ALL_WORKLOADS: dict[str, list[str]] = {
    "short": SHORT_UTTERANCES,
    "medium": CONVERSATIONAL_MEDIUM,
    "long": LITERARY_LONG,
    "numeric": NUMERIC_HEAVY,
    "time": TIME_HEAVY,
    "dates": DATE_HEAVY,
    "diacriticless": DIACRITICLESS,
    "plain_text": PLAIN_POLISH_NO_NUM,
    "adversarial": ADVERSARIAL_STRESS,
}
