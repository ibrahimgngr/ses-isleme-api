import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from sqlmodel import SQLModel, Field, create_engine, Session
from ses_isleyici import ses_isleme_fonksiyonu
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path="Ses_isleme.env")

# Veri tabanı ayarları
sqlite_file_name = "ses_analizleri.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

# Pydantic ile gelen verinin modelini belirliyoruz
class MetinIstegi(BaseModel):
    metin: str


# Veri Tabanı modeli
class AnalizSonuc(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    metin: str
    duygu: str
    anahtar_kelimeler: str

# Tablo oluşturma fonksiyonu
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

app = FastAPI()

# Güvenlik izni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uygulama başlarken veritabanı tablosu oluşturulsun
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# Geçici ses dosyaları klasörü
UPLOAD_DIR = "temp_sesler"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dosya yükkleme ve analiz Endpointi
@app.post("/analiz-baslat")
async def analiz_basat(file: UploadFile = File(...)):
    # 1. Dosyayı kaydetme
    kayit_isim = f"{uuid.uuid4()}_{file.filename}"
    dosya_yolu = os.path.join(UPLOAD_DIR, kayit_isim)
    with open(dosya_yolu, "wb") as buffer:
        buffer.write(await file.read())

    # 2. Sesi metine çevir ve konuşmacıları ayır (ses_isleyici.py)
    islem_sonucu = ses_isleme_fonksiyonu(dosya_yolu)
    metin = islem_sonucu["metin"]
    konusmalar = str(islem_sonucu["konusmalar"])

    # 3. Gemini ile duygu analizi yap
    prompt = f"""
    Sen bir duygu analisti ve anahtar kelime çıkarıcısısın.
    Aşağıdaki metni ve konuşmacı bilgilerini analiz et.
    
    ÇIKTI FORMATI:
    Kesinlikle JSON, süslü parantez, tırnak işareti veya kod bloğu kullanma.
    Doğrudan kullanıcının okuyacağı şu şablonla yanıt ver:
    
    Özet: [Buraya konuşmanın genel özetini yaz]
    Duygu Durumu: [Buraya her konuşmacının duygu durumunu ayrı ayrı anlat]
    Anahtar Kelimeler: [Kelime1, Kelime2, Kelime3 şeklinde köşeli parantez olmadan, yan yana virgülle ayırarak yaz]

    Metin: {metin}
    Konuşmacı Bilgileri: {konusmalar}
    """
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=prompt,
    )

    #Veritabanına kaydetme (basitçe yanıtı string olarak kaydediyoruz)
    yeni_kayit = AnalizSonuc(
        metin=metin,
        duygu=response.text,
        anahtar_kelimeler="Analiz edildi"
    )

    with Session(engine) as session:
        session.add(yeni_kayit)
        session.commit()

    return {"mesaj": "Analiz tamamlandı ve veritabanına kaydedildi.", "sonuc": response.text}

@app.get("/sonuclari-gor")
def sonuclari_getir():
    with Session(engine) as session:
        # Veritabanındaki tüm kayıtları çekip döndürür
        kayitlar = session.query(AnalizSonuc).all()
        return kayitlar
    
# Sadeece metin analizi yapan Endpoint
@app.post("/duygu-analizi")
async def canli_duygu_analizi(istek: MetinIstegi):
    prompt = f"""
    Sen bir duygu analisti ve anahtar kelime çıkarıcısısın.
    Aşağıdaki anlık konuşma metnini analiz et:
    Metin: {istek.metin}
    
    Lütfen bana özet, duygu durumu ve anahtar kelimeleri ver.
    """
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=prompt,
    )

    yeni_kayit = AnalizSonuc(
        metin=istek.metin,
        duygu=response.text,
        anahtar_kelimeler="canlı-dinleme"
    )

    with Session(engine) as session:
        session.add(yeni_kayit)
        session.commit()

        return {"mesaj": "Analiz başarılı", "sonuc": response.text}