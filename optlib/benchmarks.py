'''
Created on 2015/02/23

@author: Minoru Kanemasa
'''

# NumPy, matematiksel vektor islemleri icin kullanilir.
import numpy as np

# scipy.sparse, Rotate benchmark'inda seyrek rotasyon matrisleri kurmak icin kullanilir.
import scipy.sparse


# ObjectiveFunction, benchmark fonksiyonlari icin ortak bir sarmalayici siniftir.
class ObjectiveFunction:
    # E gercek fonksiyon, dim ise problem boyutudur.
    def __init__(self, E, dim, best=None):
        # _E, cozum vektorunu alip skor uretecek asil fonksiyonu saklar.
        self._E = E

        # _dim, optimizasyon probleminin kac boyutlu oldugunu saklar.
        self._dim = dim

    # Nesne fonksiyon gibi cagrildiginda objective skoru hesaplanir.
    def __call__(self, x):
        # Verilen x vektoru asil fonksiyona gonderilir ve skor dondurulur.
        return self._E(x)

    # Problem boyutunu Firefly algoritmasina bildiren fonksiyondur.
    def dim(self):
        # Kayitli problem boyutu dondurulur.
        return self._dim


# BestInfo, benchmark'in bilinen en iyi cozumunu tutmak icin kullanilir.
class BestInfo:
    # best, benchmark'in teorik veya bilinen optimum cozum vektorudur.
    def __init__(self, best=None):
        # _best alanina bilinen optimum kaydedilir.
        self._best = best

    # Bilinen optimum cozum vektorunu dondurur.
    def best(self):
        # Eger optimum tanimliysa onu dondurur.
        if self._best:
            # Bilinen en iyi cozum vektoru dondurulur.
            return self._best

        # Optimum tanimli degilse hata firlatilir.
        else:
            # Bu benchmark icin best bilgisi yok demektir.
            raise NotImplementedError


# NameInfo, benchmark siniflarina ortak name() davranisi verir.
class NameInfo:
    # name parametresi verilirse kullanilacak ozel addir.
    def __init__(self, name=None):
        # _name alanina ozel isim kaydedilir.
        self._name = name

    # Benchmark adini dondurur.
    def name(self):
        # Ozel isim verilmisse onu kullanir.
        if self._name:
            # Ozel benchmark adi dondurulur.
            return self._name

        # Ozel isim yoksa sinif adi kullanilir.
        else:
            # Sinif adi string olarak dondurulur.
            return self.__class__.__name__


# Rotate, verilen benchmark fonksiyonunu koordinat sisteminde dondurerek zorlastirir.
class Rotate(ObjectiveFunction):
    # E dondurulecek benchmark, degree ise donus acisidir.
    def __init__(self, E, degree=20):
        # Asil benchmark'in boyutu alinir.
        dim = E.dim()

        # Derece cinsinden aci radyana cevrilir.
        alpha = degree * np.pi / 180

        # Asil benchmark nesnesi saklanir.
        self._E = E

        # Rotasyon matrisi baslangicta birim matristir.
        self._R = scipy.sparse.identity(dim)

        # Tum boyut ciftleri icin ikili rotasyonlar uygulanir.
        for i in range(dim - 1):
            # i'den sonraki her j boyutu ile bir rotasyon duzlemi kurulur.
            for j in range(i + 1, dim):
                # Bu i-j duzlemi icin seyrek matris olusturulur.
                m = scipy.sparse.identity(dim, format='lil')

                # Matrisin satirlari gezilir.
                for k in range(dim):
                    # Matrisin sutunlari gezilir.
                    for l in range(dim):
                        # i-i elemani cos(alpha) olur.
                        if (k == i) and (l == i):
                            # Rotasyon matrisinin cos bileseni yazilir.
                            m[k, l] = np.cos(alpha)

                        # i-j elemani -sin(alpha) olur.
                        elif (k == i) and (l == j):
                            # Rotasyon matrisinin negatif sin bileseni yazilir.
                            m[k, l] = -np.sin(alpha)

                        # j-i elemani sin(alpha) olur.
                        elif (k == j) and (l == i):
                            # Rotasyon matrisinin pozitif sin bileseni yazilir.
                            m[k, l] = np.sin(alpha)

                        # j-j elemani cos(alpha) olur.
                        elif (k == j) and (l == j):
                            # Rotasyon matrisinin diger cos bileseni yazilir.
                            m[k, l] = np.cos(alpha)

                        # i ve j disindaki diagonal elemanlar 1 kalir.
                        elif (k == l) and not((l == i) or (l == j)):
                            # Etkilenmeyen boyutlar aynen korunur.
                            m[k, l] = 1.0

                        # Diger tum elemanlar sifirdir.
                        else:
                            # Rotasyon matrisinde ilgili eleman sifirlanir.
                            m[k, l] = 0

                # Yeni ikili rotasyon, toplam rotasyon matrisine carpilir.
                self._R = self._R * m

    # Rotate nesnesi objective gibi cagrildiginda x once dondurulur, sonra asil benchmark'a verilir.
    def __call__(self, x):
        # x, benchmark optimumu etrafinda rotate edilir ve asil objective skoru hesaplanir.
        return self._E(self._R * (x - self._E.best()) + self._E.best())

    # Rotate benchmark'inin boyutu asil benchmark boyutuyla aynidir.
    def dim(self):
        # Asil benchmark'in dim() sonucu dondurulur.
        return self._E.dim()

    # Rotate icin bilinen optimum asil benchmark'tan gelir.
    def best(self):
        # Asil benchmark'in best bilgisi dondurulur.
        return self._E.best()

    # Rotate icin quality degeri de rotate edilmis x'in asil benchmark kalitesidir.
    def quality(self, x):
        # x rotate edilir ve asil benchmark'in quality fonksiyonuna verilir.
        return self._E.quality(self._R * (x - self._E.best()) + self._E.best())

    # Rotate benchmark adini asil adin sonuna "(Rotated)" ekleyerek verir.
    def name(self):
        # Asil benchmark adi ile Rotated etiketi birlestirilir.
        return self._E.name() + ' (Rotated)'


# BenchmarkMaker, boyutlu ve toplanabilir benchmark fonksiyonlari icin ortak siniftir.
class BenchmarkMaker(ObjectiveFunction, BestInfo, NameInfo):
    # E, tek tek boyutlarda calisan fonksiyon; dim, problem boyutu; best, bilinen optimumdur.
    def __init__(self, E, dim, best=None):
        # ObjectiveFunction baslatilir.
        ObjectiveFunction.__init__(self, E, dim)

        # BestInfo baslatilir.
        BestInfo.__init__(self, best)

        # NameInfo baslatilir.
        NameInfo.__init__(self)

    # Benchmark nesnesi fonksiyon gibi cagrildiginda objective score hesaplanir.
    def __call__(self, x):
        # _E(x) her boyut icin skor uretir; np.sum bunlari toplayip tek objective score yapar.
        return np.sum(self._E(x))

    # quality, bu basit benchmark'larda objective score ile ayni kabul edilir.
    def quality(self, x):
        # x'in objective score'u quality olarak dondurulur.
        return self.__call__(x)


# Minima fonksiyonunun tek boyut icin matematiksel hata/skor hesabidir.
def _minima(vec):
    # Firefly'in 0-1 araligindaki vektoru -5 ile 5 araligina olceklenir.
    x = vec * 10.0 - 5.0

    # Polynomial fonksiyon degeri hesaplanir.
    return x ** 4 - 16 * (x ** 2) + 5 * x


# Minima benchmark'inin 0-1 araligina olceklenmis optimum noktasi yaklasik olarak hesaplanir.
_minimabest = (-2.9035 + 5.0) / 10.0


# Rastrigin fonksiyonunun tek boyut icin matematiksel hata/skor hesabidir.
def _rastrigin(vec):
    # Firefly'in 0-1 araligindaki vektoru -5.12 ile 5.12 araligina olceklenir.
    x = vec * 10.24 - 5.12

    # Rastrigin fonksiyonunun boyut bazli degeri hesaplanir.
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + 10


# Rastrigin optimumu x=0 oldugu icin 0-1 olceginde 0.5'e denk gelir.
_rastbest = 0.5


# Minima benchmark sinifi, BenchmarkMaker uzerine kurulur.
class Minima(BenchmarkMaker):
    # dim, kac boyutlu Minima problemi kurulacagini belirler.
    def __init__(self, dim):
        # Her boyut icin ayni optimum kullanilarak BenchmarkMaker baslatilir.
        BenchmarkMaker.__init__(self, _minima, dim, best=[_minimabest] * dim)


# Rastrigin benchmark sinifi, BenchmarkMaker uzerine kurulur.
class Rastrigin(BenchmarkMaker):
    # dim, kac boyutlu Rastrigin problemi kurulacagini belirler.
    def __init__(self, dim):
        # Her boyut icin optimum 0.5 kabul edilerek BenchmarkMaker baslatilir.
        BenchmarkMaker.__init__(self, _rastrigin, dim, best=[_rastbest] * dim)


# Rosenbrock benchmark sinifi, dar vadi yapili klasik optimizasyon testidir.
class Rosenbrock(ObjectiveFunction, BestInfo, NameInfo):
    # dim problem boyutu, alpha Rosenbrock ceza katsayisidir.
    def __init__(self, dim, alpha=100.0):
        # Rosenbrock en az 2 boyut ister.
        if 1 == dim:
            # Tek boyutlu Rosenbrock desteklenmez.
            raise NotImplementedError

        # ObjectiveFunction baslatilir; asil fonksiyon __call__ icinde manuel hesaplanacaktir.
        ObjectiveFunction.__init__(self, None, dim)

        # Rosenbrock optimumu gercek olcekte 1 oldugu icin 0-1 olceginde 0.6'dir.
        BestInfo.__init__(self, [6.0 / 10.0] * dim)

        # Sinif adi name() icin hazirlanir.
        NameInfo.__init__(self)

        # Rosenbrock alpha katsayisi saklanir.
        self._alpha = alpha

    # Rosenbrock objective score hesaplar.
    def __call__(self, x):
        # 0-1 araligindaki x, -5 ile 5 araligina olceklenir.
        v = x * 10.0 - 5.0

        # Toplam hata/skor sifirdan baslatilir.
        ret = 0.0

        # Rosenbrock ard isik boyut ciftleri uzerinden hesaplanir.
        for n in range(self.dim() - 1):
            # Klasik Rosenbrock terimi toplam skora eklenir.
            ret += self._alpha * ((v[n] ** 2 - v[n + 1]) ** 2) + (1 - v[n]) ** 2

        # Toplam objective score dondurulur.
        return ret

    # Rosenbrock icin quality objective score ile aynidir.
    def quality(self, x):
        # x'in Rosenbrock objective skoru quality olarak dondurulur.
        return self.__call__(x)


# Fractal benchmark sinifi, parcali/fraktal yapida bir objective yuzeyi uretir.
class Fractal(ObjectiveFunction, BestInfo, NameInfo):
    # dim, kac boyutlu Fractal problemi kurulacagini belirler.
    def __init__(self, dim):
        # ObjectiveFunction baslatilir; asil hesap __call__ icinde yapilir.
        ObjectiveFunction.__init__(self, None, dim, [0.5] * dim)

        # Fractal optimumu 0-1 olceginde tum boyutlar icin 0.5 kabul edilir.
        BestInfo.__init__(self, [0.5] * dim)

        # Sinif adi name() icin hazirlanir.
        NameInfo.__init__(self)

        # w1, fraktal bolme katsayisinin ilk siniridir.
        self.w1 = 1.0 / 3.0

        # w2, fraktal bolme katsayisinin ikinci siniridir.
        self.w2 = 2.0 / 3.0

    # Fractal objective score hesaplar.
    def __call__(self, x):
        # Cozum 0-1 araligindan cikarsa skor 0 dondurulur.
        if min(x) < 0 or 1 < max(x):
            # Aralik disi cozum icin objective degeri 0 kabul edilir.
            return 0

        # Toplam fraktal skor sifirdan baslatilir.
        obj = 0.0

        # Cozum vektorundeki her boyut ayri ayri hesaplanir.
        for elem in x:
            # Her elemanin fraktal skoru toplama eklenir.
            obj += self._frac(elem, n=30)

        # Toplam objective score dondurulur.
        return obj

    # Fractal icin quality, optimuma ne kadar fraktal seviyede yaklasildigini verir.
    def quality(self, x):
        # Cozum 0-1 araligindan cikarsa quality 0 dondurulur.
        if min(x) < 0 or 1 < max(x):
            # Aralik disi cozum kalite olarak gecersiz kabul edilir.
            return 0

        # x'in 0.5 optimumundan en uzak bilesen mesafesi hesaplanir.
        mx = max(np.fabs(x - 0.5))

        # Fraktal seviye sayaci sifirdan baslar.
        m = 0

        # Cozum optimuma belirli fraktal araliklar icinde kaldikca seviye artar.
        while mx <= 0.5 * (self.w1 ** m):
            # Bir sonraki daha dar fraktal seviyeye gecilir.
            m += 1

        # Daha yuksek seviye daha iyi oldugu icin negatif isaretli kalite dondurulur.
        return -m + 1

    # Tek bir sayinin fraktal objective katkisini recursive olarak hesaplar.
    def _frac(self, x, n=30):
        # Recursion derinligi 1 veya daha azsa taban durum uygulanir.
        if n <= 1:
            # x orta uc bolgede ise negatif katki verir.
            if self.w1 <= x and x <= self.w2:
                # Orta bolge optimum tarafini temsil ettigi icin negatif skor dondurulur.
                return -self.w1

            # x orta bolgede degilse skor katkisi sifirdir.
            else:
                # Dis bolgeler icin katki yoktur.
                return 0.0

        # Recursion devam edecekse x'in hangi uc bolgede olduguna bakilir.
        else:
            # x ilk uc bolgedeyse sol alt probleme indirgenir.
            if x < self.w1:
                # Sol bolge katsayiyla olceklenerek recursive hesaplanir.
                return self.w1 * self._frac(3 * x, n - 1)

            # x orta uc bolgedeyse negatif taban katkisi ve recursive katki birlikte hesaplanir.
            elif x <= self.w2:
                # Orta bolge daha iyi skor verir ve kalan kisim recursive hesaplanir.
                return -self.w1 + self.w1 * self._frac(3 * x - 1, n - 1)

            # x sag uc bolgedeyse sag alt probleme indirgenir.
            else:
                # Sag bolge katsayiyla olceklenerek recursive hesaplanir.
                return self.w1 * self._frac(3 * x - 2, n - 1)
