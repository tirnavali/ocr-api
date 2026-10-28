"""CPU-efficient Turkish Topic Assignment, Keyphrase Extraction, and NER with WordBox alignment."""

from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ocr_service.domain.models import (
    BBoxPt,
    BBoxPx,
    EntityLabel,
    KeywordItem,
    NamedEntity,
    NlpMode,
    PageNlpAnalysis,
    PageOcrResult,
    TopicPrediction,
    WordBox,
)
from ocr_service.pipeline.ner_onnx import LazyOnnxBertNer, OnnxBertNer

logger = logging.getLogger(__name__)

# Turkish-aware character translation tables
_TR_LOWER_MAP = str.maketrans(
    {
        "İ": "i",
        "I": "ı",
        "Â": "a",
        "Î": "i",
        "Û": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)

_PUNCT_STRIP = ".,;:!?\"'“”‘’()[]{}<>«»-–—/\\|*~`@#$%^&+=_"

TURKISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "acaba",
        "ait",
        "ama",
        "ancak",
        "arası",
        "arasında",
        "artık",
        "asla",
        "ayrıca",
        "az",
        "bana",
        "bazı",
        "belki",
        "ben",
        "benden",
        "beni",
        "benim",
        "beri",
        "beş",
        "bile",
        "bin",
        "binaenaleyh",
        "bir",
        "biraz",
        "birçok",
        "biri",
        "birisi",
        "birkaç",
        "birşey",
        "biz",
        "bizden",
        "bize",
        "bizi",
        "bizim",
        "böyle",
        "böylece",
        "bu",
        "buna",
        "bunda",
        "bundan",
        "bunlar",
        "bunları",
        "bunların",
        "bunu",
        "bunun",
        "burada",
        "bütün",
        "çoğu",
        "çok",
        "çünkü",
        "da",
        "daha",
        "dahi",
        "dair",
        "de",
        "defa",
        "değil",
        "diğer",
        "diye",
        "doksan",
        "dokuz",
        "dolayı",
        "dolayısıyla",
        "dört",
        "edecek",
        "eden",
        "ederek",
        "edilecek",
        "edilen",
        "ediliyor",
        "edilmesi",
        "edilmiş",
        "eğer",
        "elli",
        "en",
        "etmesi",
        "etti",
        "ettiği",
        "ettiğini",
        "fakat",
        "falan",
        "filan",
        "gibi",
        "göre",
        "halen",
        "hangi",
        "hangisi",
        "hani",
        "hasebiyle",
        "hatta",
        "hem",
        "henüz",
        "hep",
        "hepsi",
        "her",
        "herhangi",
        "herkes",
        "herkesin",
        "hiç",
        "hiçbir",
        "için",
        "içinde",
        "iki",
        "ile",
        "ilgili",
        "ise",
        "işte",
        "itibaren",
        "itibariyle",
        "kadar",
        "karşı",
        "kendi",
        "kendilerine",
        "kendini",
        "kendisi",
        "kendisine",
        "kendisini",
        "kez",
        "keza",
        "ki",
        "kim",
        "kime",
        "kimi",
        "kimin",
        "kimisi",
        "kimse",
        "kırk",
        "konu",
        "konusunda",
        "lakin",
        "mamafih",
        "mı",
        "mi",
        "mu",
        "mucibince",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "nedenle",
        "nerde",
        "nerede",
        "nereye",
        "nitekim",
        "niçin",
        "niye",
        "o",
        "olan",
        "olarak",
        "oldu",
        "olduğu",
        "olduğunu",
        "olduklarını",
        "olmadı",
        "olmadığı",
        "olmak",
        "olması",
        "olmayan",
        "olmaz",
        "olsa",
        "olsun",
        "olup",
        "olur",
        "olursa",
        "oluyor",
        "on",
        "ona",
        "ondan",
        "onlar",
        "onlara",
        "onlardan",
        "onları",
        "onların",
        "onu",
        "onun",
        "orada",
        "otuz",
        "oysa",
        "öyle",
        "pek",
        "rağmen",
        "sadece",
        "sanki",
        "sekiz",
        "seksen",
        "sen",
        "senden",
        "seni",
        "senin",
        "siz",
        "sizden",
        "size",
        "sizi",
        "sizin",
        "sonra",
        "şayet",
        "şey",
        "şeyden",
        "şeyi",
        "şeyler",
        "acaba",
        "şöyle",
        "şu",
        "şuna",
        "şunda",
        "şundan",
        "şunları",
        "şunu",
        "tarafından",
        "trilyon",
        "tüm",
        "üç",
        "üzere",
        "üzerine",
        "var",
        "vardı",
        "ve",
        "veya",
        "yahut",
        "yani",
        "yapacak",
        "yapılan",
        "yapılması",
        "yapıyor",
        "yapmak",
        "yaptı",
        "yaptığı",
        "yaptığını",
        "yaptıkları",
        "yedi",
        "yerine",
        "yetmiş",
        "yine",
        "yirmi",
        "yok",
        "yoksa",
        "yüz",
        "zaten",
        "zira",
    }
)

# Common Turkish inflectional suffixes ordered longest-first for lightweight CPU stemming
_TR_SUFFIXES: tuple[str, ...] = (
    "larımızın",
    "lerimizin",
    "larınızın",
    "lerinizin",
    "larından",
    "lerinden",
    "larının",
    "lerinin",
    "larında",
    "lerinde",
    "larına",
    "lerine",
    "larını",
    "lerini",
    "larımız",
    "lerimiz",
    "larınız",
    "leriniz",
    "casına",
    "cesine",
    "সından",
    "sinden",
    "sında",
    "sinde",
    "sının",
    "sinin",
    "sunun",
    "sünün",
    "mızın",
    "mizin",
    "muzun",
    "müzün",
    "lardan",
    "lerden",
    "ların",
    "lerin",
    "larda",
    "lerde",
    "larla",
    "lerle",
    "lığı",
    "liği",
    "luğu",
    "lüğü",
    "cılar",
    "ciler",
    "cular",
    "cüler",
    "daki",
    "deki",
    "taki",
    "teki",
    "ndan",
    "nden",
    "ları",
    "leri",
    "dan",
    "den",
    "tan",
    "ten",
    "nın",
    "nin",
    "nun",
    "nün",
    "mız",
    "miz",
    "muz",
    "müz",
    "nız",
    "niz",
    "nuz",
    "nüz",
    "lar",
    "ler",
    "lık",
    "lik",
    "luk",
    "lük",
    "sız",
    "siz",
    "suz",
    "süz",
    "dır",
    "dir",
    "dur",
    "dür",
    "tır",
    "tir",
    "tur",
    "tür",
    "da",
    "de",
    "ta",
    "te",
    "ya",
    "ye",
    "yı",
    "yi",
    "yu",
    "yü",
    "na",
    "ne",
    "nı",
    "ni",
    "nu",
    "nü",
    "la",
    "le",
    "lı",
    "li",
    "lu",
    "lü",
    "ca",
    "ce",
    "ın",
    "in",
    "un",
    "ün",
)


@dataclass(frozen=True, slots=True)
class TopicDefinition:
    """Turkish document/page topic category definition with weighted stems and phrases."""

    category_id: str
    category_name: str
    keywords: dict[str, float]


TURKISH_TOPIC_TAXONOMY: tuple[TopicDefinition, ...] = (
    TopicDefinition(
        category_id="politics_parliament",
        category_name="Siyaset ve Meclis",
        keywords={
            "meclis": 2.5,
            "tbmm": 3.0,
            "büyük millet meclisi": 3.5,
            "mebus": 2.8,
            "mebusan": 2.8,
            "milletvekili": 2.5,
            "riyaset": 2.2,
            "reisicumhur": 2.8,
            "cumhurbaşkan": 2.5,
            "başvekil": 2.5,
            "başbakan": 2.5,
            "icra vekilleri": 3.0,
            "heyet": 1.6,
            "vekalet": 2.0,
            "bakanlık": 1.8,
            "fırka": 2.4,
            "parti": 2.2,
            "intihap": 2.4,
            "seçim": 2.2,
            "celse": 2.5,
            "oturum": 2.0,
            "zabıt": 2.4,
            "tutanak": 2.2,
            "müzakere": 2.2,
            "cumhuriyet": 2.0,
            "egemenlik": 2.0,
            "hakimiyet": 2.0,
            "hükümet": 2.2,
            "kabine": 2.2,
        },
    ),
    TopicDefinition(
        category_id="law_legislation",
        category_name="Hukuk ve Mevzuat",
        keywords={
            "kanun": 2.8,
            "yasa": 2.5,
            "madde": 2.4,
            "mevzuat": 2.5,
            "mahkeme": 2.6,
            "yargı": 2.4,
            "hüküm": 1.8,
            "kararname": 2.6,
            "nizamname": 2.6,
            "talimatname": 2.5,
            "tüzük": 2.2,
            "yönetmelik": 2.2,
            "encümen": 2.3,
            "komisyon": 2.0,
            "adliye": 2.5,
            "adalet": 2.2,
            "ceza": 2.2,
            "hukuk": 2.5,
            "layiha": 2.6,
            "tasarı": 2.2,
            "teklif": 1.8,
            "teşkilatı esasiye": 3.2,
            "anayasa": 2.8,
            "resmi gazete": 2.8,
            "ceridei resmiye": 3.0,
            "müeyyide": 2.2,
            "temyiz": 2.4,
            "hakim": 2.0,
            "savcı": 2.0,
            "müddeiumumi": 2.6,
        },
    ),
    TopicDefinition(
        category_id="economy_finance",
        category_name="Ekonomi, Maliye ve Ticaret",
        keywords={
            "maliye": 2.8,
            "iktisat": 2.8,
            "ekonomi": 2.6,
            "bütçe": 2.8,
            "muvazenei umumiye": 3.2,
            "vergi": 2.6,
            "rüsum": 2.6,
            "gümrük": 2.5,
            "ticaret": 2.5,
            "banka": 2.5,
            "sermaye": 2.4,
            "kredi": 2.2,
            "borç": 2.0,
            "düyun": 2.8,
            "lira": 2.0,
            "kuruş": 2.0,
            "akçe": 2.2,
            "tahsisat": 2.6,
            "ödenek": 2.4,
            "ihracat": 2.5,
            "ithalat": 2.5,
            "sanayi": 2.4,
            "fabrika": 2.2,
            "üretim": 2.0,
            "fiyat": 2.0,
            "piyasa": 2.2,
            "borsa": 2.4,
            "ihale": 2.2,
            "varidat": 2.6,
            "masarif": 2.6,
        },
    ),
    TopicDefinition(
        category_id="military_defense",
        category_name="Askerî, Harp ve Güvenlik",
        keywords={
            "asker": 2.5,
            "ordu": 2.6,
            "harp": 2.8,
            "savaş": 2.6,
            "müdafaa": 2.6,
            "müdafaai milliye": 3.2,
            "savunma": 2.4,
            "cephe": 2.6,
            "fırka": 1.8,
            "kolordu": 2.8,
            "alay": 2.2,
            "tabur": 2.4,
            "bölük": 2.0,
            "kumandan": 2.6,
            "komutan": 2.5,
            "başkumandan": 3.0,
            "erkânıharbiye": 3.0,
            "genelkurmay": 2.8,
            "silah": 2.2,
            "mühimmat": 2.6,
            "taarruz": 2.6,
            "muharebe": 2.8,
            "zafer": 2.2,
            "şehit": 2.2,
            "gazi": 2.2,
            "jandarma": 2.4,
            "emniyet": 2.2,
            "asayiş": 2.5,
            "seferberlik": 2.8,
        },
    ),
    TopicDefinition(
        category_id="foreign_affairs",
        category_name="Dış İlişkiler ve Diplomasi",
        keywords={
            "hariciye": 2.8,
            "dışişleri": 2.8,
            "muahede": 3.0,
            "antlaşma": 2.8,
            "ahidname": 2.8,
            "diplomasi": 2.6,
            "sefir": 2.8,
            "büyükelçi": 2.6,
            "elçi": 2.2,
            "konsolos": 2.4,
            "murahhas": 2.8,
            "delege": 2.4,
            "konferans": 2.4,
            "lozan": 2.8,
            "mütareke": 2.8,
            "sulh": 2.4,
            "barış": 2.2,
            "nota": 2.2,
            "devletler": 2.0,
            "milletler cemiyeti": 3.0,
            "ittifak": 2.4,
            "müttefik": 2.4,
            "hudut": 2.4,
            "sınır": 2.0,
        },
    ),
    TopicDefinition(
        category_id="education_culture",
        category_name="Eğitim, Kültür ve Basın",
        keywords={
            "maarif": 3.0,
            "eğitim": 2.6,
            "öğretim": 2.5,
            "mektep": 2.8,
            "okul": 2.4,
            "medrese": 2.6,
            "darülfünun": 3.0,
            "üniversite": 2.6,
            "muallim": 2.8,
            "öğretmen": 2.5,
            "talebe": 2.6,
            "öğrenci": 2.4,
            "kültür": 2.4,
            "matbuat": 2.8,
            "basın": 2.4,
            "gazete": 2.4,
            "mecmua": 2.4,
            "dergi": 2.2,
            "kütüphane": 2.5,
            "müze": 2.4,
            "sanat": 2.2,
            "edebiyat": 2.4,
            "tarih": 1.8,
            "harf": 2.0,
            "alfabe": 2.4,
        },
    ),
    TopicDefinition(
        category_id="infrastructure_transport",
        category_name="Bayındırlık, Ulaşım ve İmar",
        keywords={
            "nafia": 3.0,
            "bayındırlık": 2.8,
            "demiryolu": 2.8,
            "şamendifer": 3.0,
            "şose": 2.6,
            "köprü": 2.4,
            "liman": 2.5,
            "inşaat": 2.4,
            "imar": 2.5,
            "ulaşım": 2.4,
            "nakliyat": 2.5,
            "telgraf": 2.5,
            "posta": 2.2,
            "telefon": 2.2,
            "elektrik": 2.2,
            "su": 1.5,
            "kanal": 2.2,
            "hat": 1.8,
            "istasyon": 2.2,
            "vapur": 2.4,
        },
    ),
    TopicDefinition(
        category_id="health_social",
        category_name="Sağlık ve Sosyal Yardım",
        keywords={
            "sıhhiye": 3.0,
            "sağlık": 2.6,
            "hastane": 2.6,
            "memleket hastanesi": 2.8,
            "tabip": 2.6,
            "doktor": 2.4,
            "hekim": 2.4,
            "hastalık": 2.4,
            "salgın": 2.6,
            "sıtma": 2.6,
            "verem": 2.6,
            "aşı": 2.4,
            "hilaliahmer": 3.0,
            "kızılay": 2.8,
            "muaveneti içtimaiye": 3.2,
            "sosyal yardım": 2.6,
            "muhacir": 2.8,
            "mübadele": 2.8,
            "iskan": 2.6,
            "yetim": 2.4,
            "himayei etfal": 3.0,
        },
    ),
    TopicDefinition(
        category_id="agriculture_land",
        category_name="Tarım, Orman ve Toprak",
        keywords={
            "ziraat": 3.0,
            "tarım": 2.6,
            "çiftçi": 2.6,
            "köylü": 2.4,
            "toprak": 2.2,
            "arazi": 2.4,
            "orman": 2.6,
            "maden": 2.5,
            "محصول": 2.4,
            "mahsul": 2.6,
            "hasat": 2.5,
            "buğday": 2.4,
            "hububat": 2.6,
            "pamuk": 2.2,
            "tütün": 2.4,
            "hayvancılık": 2.5,
            "aşar": 2.8,
            "tohum": 2.4,
        },
    ),
    TopicDefinition(
        category_id="local_administration",
        category_name="Yerel Yönetim ve İdare",
        keywords={
            "dahiliye": 2.8,
            "içişleri": 2.6,
            "vilayet": 2.6,
            "il": 1.6,
            "vali": 2.6,
            "mutasarrıf": 2.8,
            "kaymakam": 2.6,
            "kaza": 2.2,
            "ilçe": 2.2,
            "sancak": 2.4,
            "liva": 2.4,
            "nahiye": 2.4,
            "bucak": 2.0,
            "belediye": 2.6,
            "muhtar": 2.4,
            "köy": 2.0,
            "mahalle": 2.0,
            "ahali": 2.2,
            "nüfus": 2.2,
        },
    ),
)

# Turkish cities, historical provinces, and major countries for LOC NER
TURKISH_LOCATIONS: frozenset[str] = frozenset(
    {
        "adana",
        "adıyaman",
        "afyon",
        "afyonkarahisar",
        "ağrı",
        "aksaray",
        "amasya",
        "ankara",
        "antalya",
        "ardahan",
        "artvin",
        "aydın",
        "balıkesir",
        "bartın",
        "batman",
        "bayburt",
        "bilecik",
        "bingöl",
        "bitlis",
        "bolu",
        "burdur",
        "bursa",
        "çanakkale",
        "çankırı",
        "çorum",
        "denizli",
        "diyarbakır",
        "diyarbekir",
        "düzce",
        "edirne",
        "elazığ",
        "elaziz",
        "erzincan",
        "erzurum",
        "eskişehir",
        "gaziantep",
        "ayıntap",
        "giresun",
        "gümüşhane",
        "hakkari",
        "hatay",
        "antakya",
        "ığdır",
        "ısparta",
        "içel",
        "mersin",
        "istanbul",
        "dersaadet",
        "izmir",
        "kahramanmaraş",
        "maraş",
        "karabük",
        "karaman",
        "kars",
        "kastamonu",
        "kayseri",
        "kırıkkale",
        "kırklareli",
        "kırşehir",
        "kilis",
        "kocaeli",
        "izmit",
        "konya",
        "kütahya",
        "malatya",
        "manisa",
        "saruhan",
        "mardin",
        "muğla",
        "menteşe",
        "muş",
        "nevşehir",
        "niğde",
        "ordu",
        "osmaniye",
        "rize",
        "sakarya",
        "adapazarı",
        "samsun",
        "siirt",
        "sinop",
        "sivas",
        "şanlıurfa",
        "urfa",
        "şırnak",
        "tekirdağ",
        "tokat",
        "trabzon",
        "tunceli",
        "dersim",
        "uşak",
        "van",
        "yalova",
        "yozgat",
        "zonguldak",
        "anadolu",
        "rumeli",
        "trakya",
        "türkiye",
        "musul",
        "kerkük",
        "halep",
        "şam",
        "batum",
        "selanik",
        "üsküp",
        "manastır",
        "yanya",
        "işkodra",
        "kıbrıs",
        "rodos",
        "midilli",
        "sakız",
        "lozan",
        "cenevre",
        "paris",
        "londra",
        "berlin",
        "moskova",
        "roma",
        "atina",
        "sofya",
        "bükreş",
        "tahran",
        "bakü",
        "tiflis",
        "kahire",
        "ingiltere",
        "fransa",
        "almanya",
        "italya",
        "rusya",
        "sovyetler",
        "yunanistan",
        "bulgaristan",
        "iran",
        "ırak",
        "suriye",
        "mısır",
        "amerika",
        "avrupa",
        "asya",
    }
)

# Location suffix markers when preceded by a capitalized proper noun
_LOC_HEAD_NOUNS: frozenset[str] = frozenset(
    {
        "vilayeti",
        "vilayet",
        "sancağı",
        "sancak",
        "kazası",
        "kaza",
        "nahiyesi",
        "nahiye",
        "livası",
        "liva",
        "ili",
        "ilçesi",
        "belediyesi",
        "köyü",
        "mahallesi",
        "caddesi",
        "meydanı",
        "boğazı",
        "körfezi",
        "nehri",
        "dağı",
        "ovası",
    }
)

# Organization head nouns / suffixes when preceded by capitalized tokens
_ORG_HEAD_NOUNS: frozenset[str] = frozenset(
    {
        "meclisi",
        "meclis",
        "bakanlığı",
        "bakanlık",
        "vekaleti",
        "vekâleti",
        "vekalet",
        "heyeti",
        "heyet",
        "encümeni",
        "encümen",
        "komisyonu",
        "komisyon",
        "müdürlüğü",
        "müdürlük",
        "müdiriyeti",
        "riyaseti",
        "başkanlığı",
        "kumandanlığı",
        "komutanlığı",
        "bankası",
        "cemiyeti",
        "derneği",
        "vakfı",
        "fırkası",
        "partisi",
        "mahkemesi",
        "üniversitesi",
        "darülfünunu",
        "mektebi",
        "lisesi",
        "enstitüsü",
        "kurumu",
        "teşkilatı",
        "şirketi",
        "idaresi",
        "matbaası",
        "gazetesi",
        "ajansı",
    }
)

_CANONICAL_ORGS: dict[str, str] = {
    "türkiye büyük millet meclisi": "Türkiye Büyük Millet Meclisi",
    "büyük millet meclisi": "Büyük Millet Meclisi",
    "tbmm": "TBMM",
    "icra vekilleri heyeti": "İcra Vekilleri Heyeti",
    "bakanlar kurulu": "Bakanlar Kurulu",
    "cumhuriyet halk fırkası": "Cumhuriyet Halk Fırkası",
    "cumhuriyet halk partisi": "Cumhuriyet Halk Partisi",
    "müdafaa-i hukuk cemiyeti": "Müdafaa-i Hukuk Cemiyeti",
    "anadolu ve rumeli müdafaa-i hukuk cemiyeti": "Anadolu ve Rumeli Müdafaa-i Hukuk Cemiyeti",
    "hilal-i ahmer": "Hilâl-i Ahmer Cemiyeti",
    "hilaliahmer": "Hilâl-i Ahmer Cemiyeti",
    "himaye-i etfal": "Himaye-i Etfal Cemiyeti",
    "düyun-u umumiye": "Düyun-u Umumiye",
    "milletler cemiyeti": "Milletler Cemiyeti",
    "anadolu ajansı": "Anadolu Ajansı",
    "resmi gazete": "Resmî Gazete",
    "ceride-i resmiye": "Ceride-i Resmiye",
    "osmanlı bankası": "Osmanlı Bankası",
    "ziraat bankası": "Ziraat Bankası",
    "iş bankası": "Türkiye İş Bankası",
    "merkez bankası": "Türkiye Cumhuriyet Merkez Bankası",
    "erkân-ı harbiye-i umumiye": "Erkân-ı Harbiye-i Umumiye",
    "genelkurmay başkanlığı": "Genelkurmay Başkanlığı",
}

# Person honorifics and titles (pre-titles and post-titles)
_PER_PRE_TITLES: frozenset[str] = frozenset(
    {
        "gazi",
        "mareşal",
        "müşir",
        "ferik",
        "mirliva",
        "mirimiran",
        "miralay",
        "kaymakam",
        "binbaşı",
        "yüzbaşı",
        "mülazım",
        "teğmen",
        "general",
        "orgeneral",
        "korgeneral",
        "tümgeneral",
        "tuğgeneral",
        "albay",
        "yarbay",
        "doktor",
        "dr",
        "profesör",
        "prof",
        "müderris",
        "avukat",
        "mühendis",
        "reisicumhur",
        "cumhurbaşkanı",
        "başvekil",
        "başbakan",
        "vekil",
        "bakan",
        "mebus",
        "milletvekili",
        "vali",
        "sefir",
        "büyükelçi",
        "sayın",
        "bay",
        "bayan",
        "sultan",
        "şehzade",
        "halife",
        "şeyh",
        "hoca",
        "hacı",
        "hafız",
    }
)

_PER_POST_TITLES: frozenset[str] = frozenset(
    {
        "paşa",
        "bey",
        "efendi",
        "hanım",
        "hanımefendi",
        "beyefendi",
        "ağa",
        "çavuş",
        "hazretleri",
    }
)

# Turkish and Ottoman/Rumi month names -> month number (01..12)
_TURKISH_MONTHS: dict[str, int] = {
    "ocak": 1,
    "kânunusani": 1,
    "kanunusani": 1,
    "şubat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "eylül": 9,
    "ekim": 10,
    "teşrinievvel": 10,
    "kasım": 11,
    "teşrinisani": 11,
    "aralık": 12,
    "kânunuevvel": 12,
    "kanunuevvel": 12,
}


@dataclass(slots=True)
class _PageToken:
    """Internal token linked to its index in PageOcrResult.words."""

    index: int
    raw: str
    clean: str
    lower: str
    stem: str
    is_capitalized: bool
    word_box: WordBox | None = None


class TurkishMorphNormalizer:
    """Turkish Unicode NFC, case-folding, and lightweight suffix stemmer."""

    @staticmethod
    def to_lower_tr(text: str) -> str:
        """Convert Turkish string to lowercase preserving dotless 'ı' and dotted 'i'."""
        nfc = unicodedata.normalize("NFC", text)
        return nfc.translate(_TR_LOWER_MAP).lower()

    @staticmethod
    def clean_token(token: str) -> str:
        """Strip outer punctuation and normalize apostrophes."""
        nfc = unicodedata.normalize("NFC", token).strip(_PUNCT_STRIP)
        return nfc.replace("’", "'").replace("‘", "'").replace("`", "'")

    @classmethod
    def stem(cls, token: str) -> str:
        """Return lightweight morphological stem of a Turkish word."""
        cleaned = cls.clean_token(token)
        if not cleaned:
            return ""
        lower = cls.to_lower_tr(cleaned)
        # If word has apostrophe suffix (e.g., "Ankara'da", "TBMM'nin"), strip after apostrophe
        if "'" in lower:
            base, _, _ = lower.partition("'")
            if len(base) >= 2:
                return base

        # Strip longest matching inflectional suffix (up to 2 agglutinative passes)
        current = lower
        for _ in range(2):
            matched_suffix = False
            for suffix in _TR_SUFFIXES:
                min_root = 4 if suffix in ("ın", "in", "un", "ün", "la", "le", "ca", "ce") else 3
                if current.endswith(suffix) and len(current) - len(suffix) >= min_root:
                    current = current[: -len(suffix)]
                    matched_suffix = True
                    break
            if not matched_suffix:
                break
        return current


def _union_bbox_px(boxes: list[BBoxPx]) -> BBoxPx | None:
    if not boxes:
        return None
    left = min(b.left for b in boxes)
    top = min(b.top for b in boxes)
    right = max(b.right for b in boxes)
    bottom = max(b.bottom for b in boxes)
    return BBoxPx(left=left, top=top, width=max(0.0, right - left), height=max(0.0, bottom - top))


def _union_bbox_pt(boxes: list[BBoxPt]) -> BBoxPt | None:
    if not boxes:
        return None
    x = min(b.x for b in boxes)
    y = min(b.y for b in boxes)
    right = max(b.right for b in boxes)
    top = max(b.top for b in boxes)
    return BBoxPt(x=x, y=y, width=max(0.0, right - x), height=max(0.0, top - y))


class TurkishNlpAnalyzer:
    """CPU-efficient Turkish Topic Classifier, Keyphrase Extractor, and Hybrid NER Engine."""

    def __init__(self, onnx_ner: LazyOnnxBertNer | None = None) -> None:
        self._normalizer = TurkishMorphNormalizer()
        # nlp_mode=onnx uses this BERTurk model for PER/ORG/LOC; without a model directory
        # (NER_MODEL_DIR) or when it fails to load, the rule engine answers and the page
        # reports engine_used="hybrid_fast_tr" so callers can see which engine ran.
        self._onnx_ner = onnx_ner

    def analyze_page(
        self,
        text: str,
        words: list[WordBox],
        *,
        page_height_px: int = 0,
        mode: NlpMode = NlpMode.FAST,
        custom_topics: list[str] | None = None,
    ) -> PageNlpAnalysis:
        """Analyze a single page's NFC words and text for Topics, Keywords, and Named Entities."""
        if mode == NlpMode.OFF:
            return PageNlpAnalysis(engine_used="off")

        start_ns = time.perf_counter_ns()
        model = self._onnx_ner.get() if (mode == NlpMode.ONNX and self._onnx_ner) else None
        engine_name = "hybrid_onnx_tr" if model is not None else "hybrid_fast_tr"
        tokens = self._build_tokens(text, words)
        if not tokens:
            return PageNlpAnalysis(engine_used=engine_name, processing_duration_ms=0)

        topics = self._classify_topics(tokens, custom_topics=custom_topics)
        keywords = self._extract_keywords(tokens, page_height_px=page_height_px)
        entities = self._extract_entities(tokens)
        if model is not None:
            entities = self._merge_model_entities(tokens, entities, model)

        duration_ms = int((time.perf_counter_ns() - start_ns) / 1_000_000)

        return PageNlpAnalysis(
            engine_used=engine_name,
            processing_duration_ms=duration_ms,
            topics=topics,
            keywords=keywords,
            entities=entities,
        )

    def summarize_document(self, page_results: list[PageOcrResult]) -> dict[str, Any]:
        """Aggregate page-level topics, keyphrases, and named entities across all pages."""
        topic_scores: Counter[str] = Counter()
        topic_names: dict[str, str] = {}
        topic_terms: dict[str, set[str]] = {}
        keyword_scores: Counter[str] = Counter()
        entity_map: dict[tuple[str, str], dict[str, Any]] = {}

        analyzed_pages = 0
        for page in page_results:
            if not page.nlp:
                continue
            analyzed_pages += 1
            for t in page.nlp.topics:
                topic_scores[t.category_id] += t.confidence
                topic_names[t.category_id] = t.category
                topic_terms.setdefault(t.category_id, set()).update(t.matched_terms)
            for kw in page.nlp.keywords:
                keyword_scores[kw.term] += kw.score
            for ent in page.nlp.entities:
                key = (ent.label.value, TurkishMorphNormalizer.to_lower_tr(ent.normalized_text))
                if key not in entity_map:
                    entity_map[key] = {
                        "text": ent.normalized_text,
                        "label": ent.label.value,
                        "count": 0,
                        "pages": set(),
                        "max_confidence": 0.0,
                    }
                entry = entity_map[key]
                entry["count"] += 1
                entry["pages"].add(page.page_number)
                entry["max_confidence"] = max(entry["max_confidence"], ent.confidence)

        denom = max(1, analyzed_pages)
        doc_topics = [
            {
                "category_id": cid,
                "category": topic_names[cid],
                "confidence": round(min(0.99, score / denom), 3),
                "matched_terms": sorted(topic_terms.get(cid, set()))[:8],
            }
            for cid, score in topic_scores.most_common(5)
        ]

        doc_keywords = [
            {"term": term, "score": round(min(0.99, score / denom), 3)}
            for term, score in keyword_scores.most_common(12)
        ]

        sorted_entities = sorted(
            entity_map.values(),
            key=lambda item: (item["count"], item["max_confidence"]),
            reverse=True,
        )
        doc_entities = [
            {
                "text": item["text"],
                "label": item["label"],
                "mention_count": item["count"],
                "pages": sorted(item["pages"]),
                "confidence": round(item["max_confidence"], 3),
            }
            for item in sorted_entities[:30]
        ]

        return {
            "analyzed_page_count": analyzed_pages,
            "document_topics": doc_topics,
            "document_keywords": doc_keywords,
            "top_entities": doc_entities,
        }

    def _build_tokens(self, text: str, words: list[WordBox]) -> list[_PageToken]:
        tokens: list[_PageToken] = []
        if words:
            for idx, wb in enumerate(words):
                clean = self._normalizer.clean_token(wb.text)
                if not clean:
                    continue
                lower = self._normalizer.to_lower_tr(clean)
                stem = self._normalizer.stem(clean)
                is_cap = bool(clean[0].isupper())
                tokens.append(
                    _PageToken(
                        index=idx,
                        raw=wb.text,
                        clean=clean,
                        lower=lower,
                        stem=stem,
                        is_capitalized=is_cap,
                        word_box=wb,
                    )
                )
            return tokens

        # Fallback when page used existing PDF text layer without WordBox array
        for idx, raw_tok in enumerate(text.split()):
            clean = self._normalizer.clean_token(raw_tok)
            if not clean:
                continue
            lower = self._normalizer.to_lower_tr(clean)
            stem = self._normalizer.stem(clean)
            is_cap = bool(clean[0].isupper())
            tokens.append(
                _PageToken(
                    index=idx,
                    raw=raw_tok,
                    clean=clean,
                    lower=lower,
                    stem=stem,
                    is_capitalized=is_cap,
                    word_box=None,
                )
            )
        return tokens

    def _classify_topics(
        self,
        tokens: list[_PageToken],
        custom_topics: list[str] | None = None,
    ) -> list[TopicPrediction]:
        stem_counts: Counter[str] = Counter(t.stem for t in tokens if len(t.stem) >= 2)
        lower_counts: Counter[str] = Counter(t.lower for t in tokens if len(t.lower) >= 2)

        # Build bigrams and trigrams in lowercase & stem form for multi-word phrase matching
        bigrams_lower = Counter(
            f"{tokens[i].lower} {tokens[i + 1].lower}" for i in range(len(tokens) - 1)
        )
        bigrams_stem = Counter(
            f"{tokens[i].stem} {tokens[i + 1].stem}" for i in range(len(tokens) - 1)
        )
        trigrams_lower = Counter(
            f"{tokens[i].lower} {tokens[i + 1].lower} {tokens[i + 2].lower}"
            for i in range(len(tokens) - 2)
        )

        taxonomy = list(TURKISH_TOPIC_TAXONOMY)
        if custom_topics:
            for custom in custom_topics:
                c_clean = custom.strip()
                if not c_clean:
                    continue
                c_lower = self._normalizer.to_lower_tr(c_clean)
                c_stem = self._normalizer.stem(c_clean)
                kw_map = {c_lower: 3.0}
                if c_stem and c_stem != c_lower:
                    kw_map[c_stem] = 2.5
                for sub in c_lower.split():
                    if len(sub) >= 3 and sub not in TURKISH_STOPWORDS:
                        kw_map[self._normalizer.stem(sub)] = 2.0
                taxonomy.append(
                    TopicDefinition(
                        category_id=f"custom_{ re.sub(r'[^a-z0-9]+', '_', c_lower).strip('_') }",
                        category_name=c_clean,
                        keywords=kw_map,
                    )
                )

        raw_scores: list[tuple[TopicDefinition, float, list[str]]] = []
        total_words = max(10, len(tokens))
        length_norm = math.log10(total_words + 10.0)

        for topic_def in taxonomy:
            score = 0.0
            matched: list[str] = []
            for term, weight in topic_def.keywords.items():
                words_in_term = term.split()
                hits = 0
                if len(words_in_term) == 1:
                    hits = lower_counts.get(term, 0) + stem_counts.get(term, 0)
                    # Prefix stem match for compound Ottoman/Turkish roots >= 5 chars
                    if hits == 0 and len(term) >= 5:
                        hits = sum(
                            cnt
                            for st, cnt in stem_counts.items()
                            if st.startswith(term)
                        )
                elif len(words_in_term) == 2:
                    hits = bigrams_lower.get(term, 0) + bigrams_stem.get(term, 0)
                elif len(words_in_term) == 3:
                    hits = trigrams_lower.get(term, 0)

                if hits > 0:
                    # Sublinear TF dampening: 1 + ln(hits)
                    term_score = weight * (1.0 + math.log(hits))
                    score += term_score
                    matched.append(term)

            if score > 0.0:
                raw_scores.append((topic_def, score / length_norm, matched))

        if not raw_scores:
            return []

        raw_scores.sort(key=lambda item: item[1], reverse=True)
        max_score = raw_scores[0][1]

        predictions: list[TopicPrediction] = []
        for topic_def, norm_score, matched in raw_scores[:3]:
            # Calibrate confidence to [0.25, 0.98]
            relative = norm_score / max(max_score, 1e-6)
            absolute_factor = 1.0 - math.exp(-norm_score / 2.2)
            confidence = min(0.98, max(0.25, 0.45 * relative + 0.55 * absolute_factor))
            if confidence >= 0.30:
                predictions.append(
                    TopicPrediction(
                        category_id=topic_def.category_id,
                        category=topic_def.category_name,
                        confidence=round(confidence, 3),
                        matched_terms=matched[:6],
                    )
                )
        return predictions

    def _extract_keywords(
        self,
        tokens: list[_PageToken],
        page_height_px: int = 0,
    ) -> list[KeywordItem]:
        """Extract keyphrases using position-aware n-gram scoring (YAKE/TF hybrid)."""
        phrase_scores: Counter[str] = Counter()
        phrase_display: dict[str, str] = {}

        def _is_candidate_word(tok: _PageToken) -> bool:
            if len(tok.clean) < 3 or len(tok.stem) < 3:
                return False
            if tok.lower in TURKISH_STOPWORDS or tok.stem in TURKISH_STOPWORDS:
                return False
            if tok.clean.isdigit():
                return False
            # Require at least 2 alphabetic characters
            alpha_count = sum(1 for c in tok.clean if c.isalpha())
            return alpha_count >= 3

        n = len(tokens)
        for i, tok in enumerate(tokens):
            if not _is_candidate_word(tok):
                continue

            # Position boost: words near the top of the page (headlines) get 1.35x boost
            pos_boost = 1.0
            if tok.word_box and page_height_px > 0:
                rel_y = tok.word_box.bbox_source_px.top / max(1.0, float(page_height_px))
                if rel_y < 0.22:
                    pos_boost = 1.35
            elif i < max(15, n // 6):
                pos_boost = 1.25

            cap_boost = 1.15 if tok.is_capitalized else 1.0

            # Unigram score (keyed by stem to merge inflected forms)
            uni_key = tok.stem
            phrase_scores[uni_key] += 1.0 * pos_boost * cap_boost
            if uni_key not in phrase_display:
                phrase_display[uni_key] = tok.lower

            # Bigram candidate
            if i + 1 < n and _is_candidate_word(tokens[i + 1]):
                nxt = tokens[i + 1]
                # Only form bigram if on the same or adjacent line
                same_region = True
                if tok.word_box and nxt.word_box:
                    same_region = abs(tok.word_box.line_num - nxt.word_box.line_num) <= 1
                if same_region:
                    bi_key = f"{tok.stem} {nxt.stem}"
                    phrase_scores[bi_key] += 1.65 * pos_boost * cap_boost
                    if bi_key not in phrase_display:
                        phrase_display[bi_key] = f"{tok.lower} {nxt.lower}"

        if not phrase_scores:
            return []

        top_items = phrase_scores.most_common(12)
        max_sc = top_items[0][1]
        keywords: list[KeywordItem] = []
        seen_displays: set[str] = set()

        for key, sc in top_items:
            disp = phrase_display.get(key, key)
            if disp in seen_displays:
                continue
            seen_displays.add(disp)
            norm_sc = min(0.98, max(0.20, sc / max(max_sc, 1.0)))
            keywords.append(KeywordItem(term=disp, score=round(norm_sc, 3)))
            if len(keywords) >= 8:
                break

        return keywords

    def _merge_model_entities(
        self, tokens: list[_PageToken], rule_entities: list[NamedEntity], model: OnnxBertNer
    ) -> list[NamedEntity]:
        """BERTurk PER/ORG/LOC plus the rule engine's DATE and LAW_REF (the model has neither)."""
        kept = [e for e in rule_entities if e.label in (EntityLabel.DATE, EntityLabel.LAW_REF)]
        taken = {i for e in kept for i in e.word_indices}
        tags = model.tag_words([t.raw for t in tokens])

        spans: list[tuple[int, int, str, list[float]]] = []
        for i, wt in enumerate(tags):
            if wt.tag == "O" or "-" not in wt.tag:
                continue
            prefix, kind = wt.tag.split("-", 1)
            cur = spans[-1] if spans else None
            # I- continues the open span of the same kind; anything else starts a new span
            # (also a stray I- without a B-, which BERT taggers emit occasionally).
            if prefix == "I" and cur is not None and cur[1] == i and cur[2] == kind:
                spans[-1] = (cur[0], i + 1, kind, [*cur[3], wt.prob])
            else:
                spans.append((i, i + 1, kind, [wt.prob]))

        labels = {"PER": EntityLabel.PER, "ORG": EntityLabel.ORG, "LOC": EntityLabel.LOC}
        model_entities: list[NamedEntity] = []
        for start, end, kind, probs in spans:
            label = labels.get(kind)
            span_tokens = tokens[start:end]
            if label is None or any(t.index in taken for t in span_tokens):
                continue
            px = [t.word_box.bbox_source_px for t in span_tokens if t.word_box is not None]
            pt = [t.word_box.bbox_pdf_pt for t in span_tokens if t.word_box is not None and t.word_box.bbox_pdf_pt]
            raw_text = " ".join(t.clean for t in span_tokens)
            model_entities.append(
                NamedEntity(
                    text=raw_text,
                    normalized_text=raw_text,
                    label=label,
                    confidence=float(sum(probs) / len(probs)),
                    source="onnx_berturk",
                    word_indices=[t.index for t in span_tokens],
                    bbox_source_px=_union_bbox_px(px),
                    bbox_pdf_pt=_union_bbox_pt(pt),
                )
            )
        merged = kept + model_entities
        merged.sort(key=lambda e: e.word_indices[0] if e.word_indices else 0)
        return merged

    def _extract_entities(self, tokens: list[_PageToken]) -> list[NamedEntity]:
        """Extract PER, ORG, LOC, DATE, and LAW_REF entities and bind them to WordBox coordinates."""
        n = len(tokens)
        if n == 0:
            return []

        consumed: set[int] = set()
        entities: list[NamedEntity] = []

        def _make_entity(
            start: int,
            end: int,
            label: EntityLabel,
            confidence: float,
            source: str,
            normalized_override: str | None = None,
        ) -> NamedEntity:
            span_tokens = tokens[start:end]
            raw_text = " ".join(t.clean for t in span_tokens)
            norm_text = normalized_override or raw_text
            word_indices = [t.index for t in span_tokens]
            px_boxes = [t.word_box.bbox_source_px for t in span_tokens if t.word_box is not None]
            pt_boxes = [
                t.word_box.bbox_pdf_pt
                for t in span_tokens
                if t.word_box is not None and t.word_box.bbox_pdf_pt is not None
            ]
            for k in range(start, end):
                consumed.add(k)
            return NamedEntity(
                text=raw_text,
                normalized_text=norm_text,
                label=label,
                confidence=confidence,
                source=source,
                word_indices=word_indices,
                bbox_source_px=_union_bbox_px(px_boxes),
                bbox_pdf_pt=_union_bbox_pt(pt_boxes),
            )

        # 1. Multi-word and single-word DATE patterns (e.g., "23 Nisan 1920", "15 Teşrinievvel 1338", "29.10.1923")
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            # Numeric date: DD.MM.YYYY or DD/MM/YYYY
            m_num_date = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$", tok.clean)
            if m_num_date:
                d, m, y = int(m_num_date.group(1)), int(m_num_date.group(2)), int(m_num_date.group(3))
                if 1 <= d <= 31 and 1 <= m <= 12 and 1200 <= y <= 2100:
                    iso = f"{y:04d}-{m:02d}-{d:02d}"
                    entities.append(_make_entity(i, i + 1, EntityLabel.DATE, 0.97, "pattern", iso))
                    continue

            # Day + Turkish/Rumi Month (+ optional Year): e.g. "23 Nisan 1920"
            if tok.clean.isdigit() and 1 <= int(tok.clean) <= 31 and i + 1 < n:
                nxt = tokens[i + 1]
                month_num = _TURKISH_MONTHS.get(nxt.lower) or _TURKISH_MONTHS.get(nxt.stem)
                if month_num:
                    day_num = int(tok.clean)
                    if i + 2 < n and re.match(r"^\d{4}$", tokens[i + 2].clean):
                        year_num = int(tokens[i + 2].clean)
                        iso = f"{year_num:04d}-{month_num:02d}-{day_num:02d}"
                        entities.append(
                            _make_entity(i, i + 3, EntityLabel.DATE, 0.98, "pattern", iso)
                        )
                    else:
                        entities.append(
                            _make_entity(i, i + 2, EntityLabel.DATE, 0.90, "pattern")
                        )
                    continue

        # 2. LAW_REF patterns (e.g. "123 sayılı Kanun", "45 numaralı kanun", "Madde 12", "Karar No 85")
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            if tok.lower in ("madde", "kanun", "karar") and i + 1 < n and tokens[i + 1].clean.isdigit():
                entities.append(_make_entity(i, i + 2, EntityLabel.LAW_REF, 0.94, "pattern"))
                continue
            if (
                tok.clean.isdigit()
                and i + 2 < n
                and tokens[i + 1].stem in ("sayı", "numara", "nolu")
                and tokens[i + 2].stem in ("kanun", "karar", "kararname", "nizamname", "yasa")
            ):
                entities.append(_make_entity(i, i + 3, EntityLabel.LAW_REF, 0.96, "pattern"))
                continue

        # 3. Canonical Multi-word Organizations (Gazetteer match up to 6 words)
        for i in range(n):
            if i in consumed:
                continue
            for span_len in range(min(6, n - i), 0, -1):
                if any((i + offset) in consumed for offset in range(span_len)):
                    continue
                phrase_lower = " ".join(tokens[i + offset].lower for offset in range(span_len))
                phrase_stem = " ".join(tokens[i + offset].stem for offset in range(span_len))
                canonical = _CANONICAL_ORGS.get(phrase_lower) or _CANONICAL_ORGS.get(phrase_stem)
                if canonical:
                    entities.append(
                        _make_entity(
                            i,
                            i + span_len,
                            EntityLabel.ORG,
                            0.97,
                            "gazetteer",
                            canonical,
                        )
                    )
                    break

        # 4. Organization Suffix Phrases (e.g. "Maliye Vekâleti", "Ankara İstiklal Mahkemesi", "Hariciye Encümeni")
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            if (tok.lower in _ORG_HEAD_NOUNS or tok.stem in _ORG_HEAD_NOUNS) and i >= 1:
                start = i
                while (
                    start - 1 >= 0
                    and (start - 1) not in consumed
                    and (i - (start - 1)) <= 4
                    and (
                        tokens[start - 1].is_capitalized
                        or tokens[start - 1].lower in ("ve", "umumiye", "milliye")
                    )
                    and tokens[start - 1].lower not in TURKISH_STOPWORDS
                ):
                    start -= 1
                if start < i:
                    entities.append(
                        _make_entity(start, i + 1, EntityLabel.ORG, 0.91, "hybrid_rule")
                    )

        # 5. Person Names with Pre-Titles or Post-Titles (e.g. "Mustafa Kemal Paşa", "Gazi Mustafa Kemal", "İsmet Paşa", "Dr. Rıza Nur Bey")
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            # Post-title anchor: [CapName1, CapName2?] + Paşa/Bey/Efendi/Hanım
            if tok.lower in _PER_POST_TITLES or tok.stem in _PER_POST_TITLES:
                start = i
                while (
                    start - 1 >= 0
                    and (start - 1) not in consumed
                    and (i - (start - 1)) <= 3
                    and tokens[start - 1].is_capitalized
                    and tokens[start - 1].lower not in TURKISH_STOPWORDS
                    and tokens[start - 1].stem not in TURKISH_LOCATIONS
                ):
                    start -= 1
                # Also include preceding pre-title if present (e.g. "Gazi Mustafa Kemal Paşa")
                if (
                    start < i
                    and start - 1 >= 0
                    and (start - 1) not in consumed
                    and tokens[start - 1].lower in _PER_PRE_TITLES
                ):
                    start -= 1
                if start < i:
                    entities.append(
                        _make_entity(start, i + 1, EntityLabel.PER, 0.95, "hybrid_rule")
                    )
                    continue

        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            # Pre-title anchor: Gazi/Doktor/Başvekil/Mebus + [CapName1, CapName2?]
            if tok.lower in _PER_PRE_TITLES and i + 1 < n:
                end = i + 1
                while (
                    end < n
                    and end not in consumed
                    and (end - i) <= 4
                    and tokens[end].is_capitalized
                    and tokens[end].lower not in TURKISH_STOPWORDS
                    and tokens[end].stem not in TURKISH_LOCATIONS
                    and tokens[end].lower not in _ORG_HEAD_NOUNS
                ):
                    end += 1
                if end > i + 1:
                    entities.append(
                        _make_entity(i, end, EntityLabel.PER, 0.92, "hybrid_rule")
                    )

        # 6. Locations: Gazetteer + Location Head Nouns (e.g. "Ankara Vilayeti", "İzmir", "Lozan")
        for i in range(n):
            if i in consumed:
                continue
            tok = tokens[i]
            if tok.is_capitalized and (
                tok.lower in TURKISH_LOCATIONS or tok.stem in TURKISH_LOCATIONS
            ):
                end = i + 1
                if (
                    end < n
                    and end not in consumed
                    and (
                        tokens[end].lower in _LOC_HEAD_NOUNS
                        or tokens[end].stem in _LOC_HEAD_NOUNS
                    )
                ):
                    end += 1
                norm_loc = tok.stem.capitalize() if end == i + 1 else None
                entities.append(
                    _make_entity(i, end, EntityLabel.LOC, 0.94, "gazetteer", norm_loc)
                )

        # Sort entities in reading order by first token index
        entities.sort(key=lambda e: e.word_indices[0] if e.word_indices else 0)
        return entities
