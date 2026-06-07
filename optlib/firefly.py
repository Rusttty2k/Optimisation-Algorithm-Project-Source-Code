
# NumPy; vektor islemleri, rastgele sayilar, norm ve ussel fonksiyonlar icin kullanilir.
import numpy as np

# threads > 1 secilirse paralel objective hesaplamak icin multiprocessing gerekir.
import multiprocessing as mp


# Firefly sinifi, klasik Firefly Algorithm mantigiyla surekli optimizasyon yapar.
class Firefly:
    '''
    Firefly Algorithm
    '''

    # Algoritmanin parametreleri burada alinir ve nesne icinde saklanir.
    def __init__(self, ite, pop=20, alpha=0.25, beta=0.5, gamma=1.0, variant='sfa', torus=True, best=None, ftarget=-np.inf, threads=1):
        '''
        Default values are stable and usually fine.
        Give maximum iteration (ite) and population size (pop) if needed based on the time you have.
        Set 1 < thread for multithreading, however, creating threads have initial cost.
        It is not recommended using multithreading for light functions.
        '''
        # ite: Her populasyon icin kac tur/iterasyonluk hesap butcesi kullanilacagini belirler.
        self.ite = ite

        # pop: Populasyondaki ates bocegi sayisidir.
        self.pop = pop

        # beta: Daha iyi cozumlere dogru cekilme katsayisidir.
        self.beta = beta

        # alpha: Rastgele hareketin buyuklugunu belirler.
        self.alpha = alpha

        # gamma: Mesafe arttikca cekimin ne kadar hizli azalacagini belirler.
        self.gamma = gamma

        # variant: Hangi Firefly hareket formulu kullanilacak bilgisidir.
        self.variant = variant

        # torus: Degerler 0-1 araligindan cikarsa tekrar bu araliga sarilsin mi bilgisidir.
        self.torus = torus

        # _best: Disaridan baslangic icin verilen en iyi cozum varsa burada saklanir.
        self._best = best

        # _ftarget: Bu skorun altina inilirse algoritma erken durabilir.
        self._ftarget = ftarget

        # threads: Objective hesaplarinda kac paralel is parcacigi kullanilacagini belirler.
        self.threads = threads

    # Populasyonu, skor listelerini ve grafik icin tutulacak gecmisi baslatir.
    def _init(self, dim, E):
        # pbests: Her ates boceginin kendi buldugu en iyi konumu tutar.
        self.pbests = [np.random.rand(dim) for x in range(self.pop)]

        # currents: Her ates boceginin su anki konumunu tutar.
        self.currents = [np.random.rand(dim) for x in range(self.pop)]

        # Eger kullanici baslangic best degeri vermediyse rastgele bir global best atanir.
        if None == self._best:
            # Global en iyi konum baslangicta rastgele secilir.
            self._best = np.random.rand(dim)

            # Skor sonsuz yapilir; boylece ilk gercek cozum mutlaka daha iyi sayilir.
            self._bestScore = np.inf

        # Eger kullanici baslangic best degeri verdiyse onun objective skoru hesaplanir.
        else:
            # Verilen best vektorunun objective degeri hesaplanir.
            self._bestScore = E(np.array(self._best))

            # Verilen best listeyse NumPy dizisine cevrilir.
            self._best = np.array(self._best)

        # pbestScores: Her ates boceginin kendi en iyi skorunu tutar.
        self.pbestScores = [np.inf for x in range(self.pop)]

        # currentScores: Her ates boceginin mevcut konum skorunu tutar.
        self.currentScores = [np.inf for x in range(self.pop)]

        # _fcall: Objective fonksiyonu kac kez cagrildi sayacidir.
        self._fcall = 0

        # changed: Konumu degisen ates boceklerinin tekrar skorlanmasini saglar.
        self.changed = [True for x in range(self.pop)]

        # history_fcalls: Convergence grafiginin x ekseni icin fonksiyon cagrisi sayilarini tutar.
        self.history_fcalls = []

        # history_scores: Convergence grafiginin y ekseni icin en iyi skor gecmisini tutar.
        self.history_scores = []

        # history_vectors: Her kayit anindaki en iyi cozum vektorunu tutar.
        self.history_vectors = []

    # O anki global en iyi sonucu grafik ve CSV ciktisi icin kaydeder.
    def _recordHistory(self):
        # O ana kadar yapilan objective fonksiyonu cagrisi sayisi kaydedilir.
        self.history_fcalls.append(self._fcall)

        # O ana kadar bulunan en iyi objective skoru kaydedilir.
        self.history_scores.append(self._bestScore)

        # O ana kadar bulunan en iyi cozum vektorunun kopyasi kaydedilir.
        self.history_vectors.append(self._best.copy())

    # Ates boceklerinin mevcut konumlari objective fonksiyonu ile skorlanir.
    def _calcObj(self, E):
        # threads 1 ise skorlar sirayla hesaplanir.
        if 1 == self.threads:
            # Populasyondaki her ates bocegi gezilir.
            for i in range(self.pop):
                # Sadece konumu degismis olan ates bocegi tekrar hesaplanir.
                if self.changed[i]:
                    # Hesaplanacagi icin changed bayragi tekrar False yapilir.
                    self.changed[i] = False

                    # Toplam hesap butcesi dolduysa donguden cikilir.
                    if self.pop * self.ite <= self._fcall:
                        # Daha fazla objective hesabi yapilmamasi icin dongu kirilir.
                        break

                    # i. ates boceginin konumu objective fonksiyonuna verilir ve skoru alinir.
                    self.currentScores[i] = E(self.currents[i])

                    # Bir objective fonksiyonu cagrisi yapildigi icin sayac artirilir.
                    self._fcall += 1

        # threads 1'den buyukse skorlar multiprocessing ile paralel hesaplanir.
        else:
            # starmap kullanimi icin her cozum vektoru tek elemanli arguman listesine sarilir.
            args = [[self.currents[k]] for k in range(self.pop)]

            # Belirtilen thread/process sayisiyla havuz olusturulur.
            with mp.Pool(self.threads) as p:
                # Objective fonksiyonu tum current vektorleri icin paralel calistirilir.
                results = p.starmap(E, args)

            # Paralel hesaplanan skorlar mevcut skor listesine yazilir.
            self.currentScores = results

            # Paralelde tum populasyon hesaplandigi icin fcall pop kadar artar.
            self._fcall += self.pop

    # Kisisel en iyi ve global en iyi cozumleri mevcut skorlara gore gunceller.
    def _rewriteBests(self, E):
        # Populasyondaki her ates bocegi kontrol edilir.
        for i in range(self.pop):
            # Mevcut skor, o ates boceginin eski en iyi skorundan iyiyse kayit yenilenir.
            if self.currentScores[i] <= self.pbestScores[i]:
                # Kisisel en iyi konum, mevcut konumun kopyasi olur.
                self.pbests[i] = self.currents[i].copy()

                # Kisisel en iyi skor, mevcut skor olur.
                self.pbestScores[i] = self.currentScores[i]

                # Kisisel en iyi skor global en iyi skordan da iyiyse global best guncellenir.
                if self.pbestScores[i] <= self._bestScore:
                    # Global en iyi cozum vektoru kaydedilir.
                    self._best = self.pbests[i].copy()

                    # Global en iyi objective skoru kaydedilir.
                    self._bestScore = self.pbestScores[i]

    # Algoritmanin ana calisma fonksiyonudur; disaridan objective fonksiyonu E alir.
    def run(self, E):
        '''
        Make sure that you rescale the objective function within 0 < x < 1,
        since the initial points are uniformly distributed within that range.
        '''
        # Problem boyutu objective nesnesinden alinir.
        dim = E.dim()

        # Populasyon, skorlar ve gecmis listeleri baslatilir.
        self._init(dim, E)

        # Baslangic populasyonunun objective skorlari hesaplanir.
        self._calcObj(E)

        # Baslangic populasyonundan kisisel/global en iyiler bulunur.
        self._rewriteBests(E)

        # Ilk best sonucu convergence gecmisine yazilir.
        self._recordHistory()

        # cnt: Hangi iki ates boceginin karsilastirilacagini sirayla secen sayactir.
        cnt = 0

        # Objective cagrisi toplam butceyi asmadigi surece algoritma devam eder.
        while self._fcall < self.pop * self.ite:
            # a indeksi, hareket ettirilecek ates bocegi adayini belirler.
            a = cnt % self.pop

            # b indeksi, karsilastirilacak diger ates bocegini belirler.
            b = int(cnt / self.pop) % self.pop

            # a ve b yer degistirir; boylece tarama sirasi kodun orijinal davranisiyla ayni kalir.
            a, b = b, a

            # t, algoritmanin 0 ile 1 arasindaki ilerleme oranidir.
            t = self._fcall / float(self.ite * self.pop)

            # beta cekim katsayisidir; bu implementasyonda self.beta yerine sabit 0.267 kullanilir.
            beta = 0.267

            # alpha zamanla azalir; baslarda kesif, sonlarda daha sakin arama yapilir.
            alpha = self.alpha * np.exp(-5.0 * t)

            # b'nin skoru a'dan daha iyi veya esitse a, b'ye dogru hareket ettirilir.
            if self.currentScores[b] <= self.currentScores[a]:
                # a ile b arasindaki Oklid mesafesi hesaplanir.
                r = np.linalg.norm(self.currents[b] - self.currents[a])

                # gamma boyuta gore normalize edilir; boyut artinca cekim asiri sertlesmesin diye kullanilir.
                gamma = self.gamma / np.sqrt(dim)

                # inside, exp icindeki mesafe cezasidir; mesafe buyudukce cekim azalir.
                inside = -(gamma * (r ** 2.0) / float(dim)) * 10

                # a ile global best arasindaki mesafe hesaplanir.
                rg = np.linalg.norm(self._best - self.currents[a])

                # inside2, global best cekimi icin exp icindeki mesafe cezasidir.
                inside2 = -(gamma * (rg ** 2.0) / float(dim)) * 10

                # sfa varyanti, a'yi b'ye dogru ceker ve Gaussian rastgelelik ekler.
                if 'sfa' == self.variant:
                    # Yeni a = eski a + b'ye dogru cekim + rastgele hareket.
                    self.currents[a] = self.currents[a] + beta * np.exp(inside) * (self.currents[b] - self.currents[a]) + alpha * np.random.randn(dim)

                # mfa1 varyanti, a'yi hem b'ye hem global best'e dogru ceker.
                elif 'mfa1' == self.variant:
                    # Yeni a = eski a + b cekimi + global best cekimi + uniform rastgele hareket.
                    self.currents[a] = self.currents[a] + beta * np.exp(inside) * (self.currents[b] - self.currents[a]) + beta * np.exp(inside2) * (self._best - self.currents[a]) + alpha * (np.random.rand(dim) - 0.5)

                # nmfa varyanti, global best etrafinda ek rastgele/yonlu bir terim daha kullanir.
                elif 'nmfa' == self.variant:
                    # lambd, global best'ten uzaklikla olceklenen ek rastgele terimin katsayisidir.
                    lambd = 2.5

                    # Yeni a = b cekimi + global best cekimi + global best etrafinda ekstra hareket + rastgele hareket.
                    self.currents[a] = self.currents[a] + beta * np.exp(inside) * (self.currents[b] - self.currents[a]) + beta * np.exp(inside2) * (self._best - self.currents[a]) + lambd * (np.random.rand(dim) - 0.5) * (self.currents[a] - self._best) + alpha * (np.random.rand(dim) - 0.5)

                # torus aciksa, cozum vektorundeki her deger tekrar 0-1 araligina sarilir.
                if self.torus:
                    # i degiskeni okunurluk icin hareket eden ates bocegi indeksine esitlenir.
                    i = a

                    # Cozum vektorundeki her boyut tek tek kontrol edilir.
                    for j in range(dim):
                        # Tam sayi kismi cikarilarak deger 0-1 bandina yaklastirilir.
                        self.currents[i][j] -= int(self.currents[i][j])

                        # Deger negatif kaldiy sa 1 eklenerek 0-1 araligina sarilir.
                        if self.currents[i][j] < 0:
                            # Negatif deger torus mantigiyla araligin son tarafina tasinir.
                            self.currents[i][j] += 1

                # Hareket eden a ates boceginin yeni objective skoru hesaplanir.
                self.currentScores[a] = E(self.currents[a])

                # Yeni objective hesabi yapildigi icin sayac artirilir.
                self._fcall += 1

                # Yeni skor, a'nin kisisel en iyi skorundan iyiyse pbest guncellenir.
                if self.currentScores[a] <= self.pbestScores[a]:
                    # a'nin kisisel en iyi vektoru yeni konum olur.
                    self.pbests[a] = self.currents[a].copy()

                    # a'nin kisisel en iyi skoru yeni skor olur.
                    self.pbestScores[a] = self.currentScores[a]

                    # Bu kisisel en iyi global en iyiden de iyiyse global best guncellenir.
                    if self.pbestScores[a] <= self._bestScore:
                        # Global en iyi vektor yeni pbest olur.
                        self._best = self.pbests[a].copy()

                        # Global en iyi skor yeni pbest skor olur.
                        self._bestScore = self.pbestScores[a]

                        # Hedef skora ulasildiysa algoritma erken durur.
                        if self._bestScore <= self._ftarget:
                            # Donguden cikarak daha fazla hesap yapilmasi engellenir.
                            break

                # Hareket ve skor guncellemesi sonrasi convergence gecmisi kaydedilir.
                self._recordHistory()

            # Bir sonraki a-b cifti icin sayac artirilir.
            cnt += 1

        # Algoritmanin buldugu en iyi cozum vektoru dondurulur.
        return self._best

    # Algoritma adini disari vermek icin yardimci fonksiyon.
    def name(self):
        # Sinifin adini string olarak dondurur.
        return self.__class__.__name__


# MathFly, Firefly sinifindan tureyen alternatif bir hareket kuralidir.
class MathFly(Firefly):
    '''
    MathFly variation
    '''

    # MathFly icin varsayilan parametreler Firefly'dan biraz farklidir.
    def __init__(self, ite, pop=20, alpha=7.72, beta=1.0, torus=True, best=None, ftarget=-np.inf, threads=1):
        '''
        Default values are stable and usually fine.
        Give maximum iteration (ite) and population size (pop) if needed based on the time you have.
        Set 1 < thread for multithreading, however, creating threads have initial cost.
        It is not recommended using multithreading for light functions.
        '''
        # ite: Toplam hesap butcesini belirleyen iterasyon sayisidir.
        self.ite = ite

        # pop: Populasyondaki cozum/ates bocegi sayisidir.
        self.pop = pop

        # beta: Secilen daha iyi cozumun yonune dogru hareket katsayisidir.
        self.beta = beta

        # alpha: Rastgele hareketin zamanla azalma hizini etkiler.
        self.alpha = alpha

        # torus: Cozumleri 0-1 araligina sarmak icin kullanilir.
        self.torus = torus

        # _best: Baslangic en iyi vektoru verilirse burada saklanir.
        self._best = best

        # _ftarget: Bu skordan daha iyi sonuc bulunursa durma hedefidir.
        self._ftarget = ftarget

        # threads: Objective hesaplarinda paralellik icin kullanilir.
        self.threads = threads

    # MathFly algoritmasinin ana calisma dongusudur.
    def run(self, E):
        '''
        Make sure that you rescale the objective function within 0 < x < 1,
        since the initial points are uniformly distributed within that range.
        '''
        # Problem boyutu objective nesnesinden alinir.
        dim = E.dim()

        # Populasyon ve yardimci listeler baslatilir.
        self._init(dim, E)

        # Baslangic populasyonunun objective skorlar i hesaplanir.
        self._calcObj(E)

        # Hesap butcesi dolana kadar MathFly hareketleri yapilir.
        while self._fcall < self.pop * self.ite:
            # Mevcut skorlara gore kisisel ve global en iyiler guncellenir.
            self._rewriteBests(E)

            # Convergence grafigi icin o anki best bilgisi kaydedilir.
            self._recordHistory()

            # Hedef skora ulasildiysa algoritma erken durur.
            if self._bestScore < self._ftarget:
                # Dongu kirilir ve mevcut en iyi cozum dondurulur.
                break

            # Populasyon bir sonraki konumlara tasinir.
            self._moveToNext(dim)

            # Yeni konumlarin objective skorlari hesaplanir.
            self._calcObj(E)

        # Dongu bittikten sonra son best guncellemesi yapilir.
        self._rewriteBests(E)

        # Son best durumu da convergence gecmisine yazilir.
        self._recordHistory()

        # Bulunan en iyi cozum vektoru dondurulur.
        return self._best

    # MathFly populasyonunu ve gecmis kayitlarini baslatir.
    def _init(self, dim, E):
        # Her bireyin kisisel en iyi vektoru baslangicta rastgele atanir.
        self.pbests = [np.random.rand(dim) for x in range(self.pop)]

        # Her bireyin mevcut konumu baslangicta rastgele atanir.
        self.currents = [np.random.rand(dim) for x in range(self.pop)]

        # Baslangic global best verilmediyse rastgele secilir.
        if None == self._best:
            # Global best vektoru rastgele baslatilir.
            self._best = np.random.rand(dim)

            # Global best skoru sonsuz yapilir.
            self._bestScore = np.inf

        # Baslangic global best verildiyse skoru hesaplanir.
        else:
            # Verilen best vektorunun objective skoru hesaplanir.
            self._bestScore = E(np.array(self._best))

            # Verilen best vektoru NumPy dizisine cevrilir.
            self._best = np.array(self._best)

        # Her bireyin kisisel en iyi skoru baslangicta sonsuzdur.
        self.pbestScores = [np.inf for x in range(self.pop)]

        # Her bireyin mevcut skoru baslangicta sonsuzdur.
        self.currentScores = [np.inf for x in range(self.pop)]

        # Objective cagrisi sayaci sifirlanir.
        self._fcall = 0

        # Tum bireyler baslangicta hesaplanacak olarak isaretlenir.
        self.changed = [True for x in range(self.pop)]

        # Function call gecmisi sifirlanir.
        self.history_fcalls = []

        # Best score gecmisi sifirlanir.
        self.history_scores = []

        # Best vector gecmisi sifirlanir.
        self.history_vectors = []

    # MathFly bireylerini daha iyi secilmis pbest'lere dogru hareket ettirir.
    def _moveToNext(self, dim):
        # t, algoritmanin 0-1 arasindaki ilerleme oranidir.
        t = self._fcall / float(self.ite * self.pop)

        # Populasyondaki her birey icin yeni konum hesaplanir.
        for i in range(self.pop):
            # q1, karsilastirma icin rastgele secilen birey indeksidir.
            q1 = np.random.randint(0, self.pop)

            # q1 daha kotu kaldigi surece yeni q1 secilir.
            while self.pbestScores[i] < self.pbestScores[q1]:
                # Daha iyi veya esit bir aday bulmak icin q1 yeniden secilir.
                q1 = np.random.randint(0, self.pop)

            # Secilen q1, i'den daha iyi veya esitse i hareket ettirilir.
            if self.pbestScores[q1] <= self.pbestScores[i]:
                # i'nin konumu degisecegi icin tekrar skorlanmasi gerekecek.
                self.changed[i] = True

                # i, q1'in pbest konumuna dogru cekilir ve azalan rastgelelik eklenir.
                self.currents[i] = self.currents[i] + self.beta * (self.pbests[q1] - self.currents[i]) + 0.254 * np.exp(-self.alpha * t) * np.random.randn(dim)

                # torus aciksa cozum vektoru tekrar 0-1 araligina sarilir.
                if self.torus:
                    # Her boyut tek tek normalize edilir.
                    for j in range(dim):
                        # Degerin tam sayi kismi atilarak araliga yaklastirilir.
                        self.currents[i][j] -= int(self.currents[i][j])

                        # Deger negatifse 1 eklenerek pozitif araliga tasinir.
                        if self.currents[i][j] < 0:
                            # Negatif deger 0-1 araliginin son tarafina sarilir.
                            self.currents[i][j] += 1

    # Algoritma adini disari vermek icin yardimci fonksiyon.
    def name(self):
        # Sinif adini string olarak dondurur.
        return self.__class__.__name__
