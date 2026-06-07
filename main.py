

# copy modulu, algoritma sablonunu her kosu icin temiz bir nesneye kopyalamak icin kullanilir.
import copy

# csv modulu, tum deney sonucunu summary.csv dosyasina yazmak icin kullanilir.
import csv

# os modulu, klasor olusturma ve dosya yolu birlestirme islemleri icin kullanilir.
import os

# NumPy, istatistik hesaplari ve vektorleri CSV'ye yazmak icin kullanilir.
import numpy as np

# matplotlib backend'i Agg yapilir; boylece ekranda pencere acmadan PNG grafik uretilir.
import matplotlib

# Agg backend'i, bu satirdan sonra import edilen pyplot tarafindan kullanilir.
matplotlib.use('Agg')

# pyplot, convergence ve cozum vektoru grafiklerini cizmek icin kullanilir.
import matplotlib.pylab as plt

# Benchmark problemleri buradan gelir: Fractal, Minima, Rastrigin, Rosenbrock.
import optlib.benchmarks

# Firefly ve Firefly varyantlari buradan gelir.
import optlib.firefly


# Dosya/klasor adlarinda sorun cikartabilecek karakterleri guvenli karaktere cevirir.
def safe_name(text):
    # Dosya adinda kalmasina izin verilen karakterler burada tanimlanir.
    allowed = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'

    # Izin verilen karakterler aynen kalir; diger karakterler alt cizgiye cevrilir.
    return ''.join(ch if ch in allowed else '_' for ch in text)


# Bir algoritma kosusunun convergence grafigini PNG olarak kaydeder.
def save_convergence_plot(path, alg, obj_name, alg_name, run_index):
    # Yeni bir figur olusturulur; boyut inch cinsindendir.
    plt.figure(figsize=(8, 5))

    # X eksenine function call, Y eksenine o ana kadarki en iyi skor yazilir.
    plt.plot(alg.history_fcalls, alg.history_scores, linewidth=2)

    # X ekseni etiketi function call olarak ayarlanir.
    plt.xlabel('Function call')

    # Y ekseni etiketi best objective score olarak ayarlanir.
    plt.ylabel('Best objective score')

    # Grafik basliginda problem, algoritma ve kosu numarasi gosterilir.
    plt.title('%s - %s - run %d' % (obj_name, alg_name, run_index))

    # Izgaralar hafif saydamlikla acilir; grafik okumasi kolaylasir.
    plt.grid(True, alpha=0.3)

    # Etiketlerin ve basligin kesilmemesi icin yerlesim sikilastirilir.
    plt.tight_layout()

    # Grafik belirtilen dosya yoluna PNG olarak kaydedilir.
    plt.savefig(path, dpi=150)

    # Figur kapatilir; cok sayida grafik uretirken bellek sismez.
    plt.close()


# Finalde bulunan en iyi cozum vektorunu boyut-deger grafigi olarak kaydeder.
def save_vector_plot(path, best, obj_name, alg_name, run_index):
    # Yeni bir figur olusturulur.
    plt.figure(figsize=(8, 5))

    # X ekseninde boyut numarasi, Y ekseninde o boyuttaki cozum degeri gosterilir.
    plt.plot(range(1, len(best) + 1), best, marker='o', linewidth=2)

    # X ekseni etiketi boyut bilgisini anlatir.
    plt.xlabel('Dimension')

    # Y ekseni etiketi cozum vektorundeki degerleri anlatir.
    plt.ylabel('Value')

    # Grafik basliginda hangi problem/algoritma/kosu oldugu yazilir.
    plt.title('Best solution vector - %s - %s - run %d' % (obj_name, alg_name, run_index))

    # Firefly bu projede cozumleri 0-1 araliginda tuttugu icin y ekseni buna gore sabitlenir.
    plt.ylim(-0.05, 1.05)

    # Izgara acilir; noktalari okumak kolaylasir.
    plt.grid(True, alpha=0.3)

    # Grafik elemanlari tasmasin diye yerlesim sikilastirilir.
    plt.tight_layout()

    # Grafik belirtilen dosya yoluna PNG olarak kaydedilir.
    plt.savefig(path, dpi=150)

    # Figur kapatilir.
    plt.close()


# Program dogrudan python main.py ile calistirildiginda deney akisi buradan baslar.
if __name__ == '__main__':
    # ite, her algoritmanin kullanacagi iterasyon/hesap butcesini belirler.
    ite = 300

    # pop, populasyondaki ates bocegi sayisidir.
    pop = 20

    # runs, her problem-algoritma kombinasyonunun kac kez tekrar edilecegidir.
    runs = 10

    # dim, benchmark problemlerinin kac boyutlu cozuldugunu belirler.
    dim = 10

    # out_dir, grafiklerin ve CSV dosyalarinin yazilacagi ana klasordur.
    out_dir = 'firefly_outputs'

    # algs listesi, karsilastirilacak Firefly algoritma ayarlarini tutar.
    algs = [
        # firefly_sfa, standart Firefly hareket kuralini kullanan ilk algoritmadir.
        ('firefly_sfa', optlib.firefly.Firefly(ite, pop, alpha=0.25, beta=0.5, gamma=2.4)),

        # firefly_nmfa, nmfa varyantini kullanan ikinci algoritmadir.
        ('firefly_nmfa', optlib.firefly.Firefly(ite, pop, alpha=0.01, beta=0.02, gamma=16.0, variant='nmfa')),
    ]

    # objs listesi, algoritmalarin uzerinde denenecegi benchmark problemlerini tutar.
    objs = [
        # Fractal problemi, cok parcali ve zor bir arama yuzeyi verir.
        optlib.benchmarks.Fractal(dim),

        # Minima problemi, polynomial bir benchmark fonksiyonudur.
        optlib.benchmarks.Minima(dim),

        # Rastrigin problemi, cok yerel minimum icerdigi icin metaheuristic testlerinde yaygindir.
        optlib.benchmarks.Rastrigin(dim),

        # Rosenbrock problemi, dar vadi yapisiyla optimizasyon algoritmalarini zorlar.
        optlib.benchmarks.Rosenbrock(dim),
    ]

    # Ana cikti klasoru yoksa olusturulur.
    os.makedirs(out_dir, exist_ok=True)

    # summary_rows, tum kosularin ozet bilgilerini bellekte toplar.
    summary_rows = []

    # Her benchmark problemi sirayla gezilir.
    for obj in objs:
        # Problem adi obj.name() ile alinir.
        obj_name = obj.name()

        # Konsola hangi problemde calisildigi yazilir.
        print('\nObjective:', obj_name)

        # Her algoritma ayari ayni problem uzerinde denenir.
        for alg_name, alg_template in algs:
            # Bu problem-algoritma ikilisi icin ozel cikti klasoru hazirlanir.
            run_dir = os.path.join(out_dir, safe_name(obj_name), safe_name(alg_name))

            # Klasor yoksa olusturulur.
            os.makedirs(run_dir, exist_ok=True)

            # qualities listesi, tekrar kosularinin quality degerlerini tutar.
            qualities = []

            # Ayni deney birden fazla kez tekrarlanir; algoritma rastgele oldugu icin bu onemlidir.
            for run_index in range(1, runs + 1):
                # Algoritma sablonu kopyalanir; onceki kosunun durumu yeni kosuya karismasin.
                alg = copy.deepcopy(alg_template)

                # Algoritma benchmark problemi uzerinde calistirilir ve en iyi vektor alinir.
                best = alg.run(obj)

                # Algoritmanin tuttugu global en iyi objective skoru okunur.
                objective_score = alg._bestScore

                # Benchmark'in quality fonksiyonu varsa final cozumun kalite degeri hesaplanir.
                quality = obj.quality(best)

                # Bu kosunun quality degeri istatistik icin listeye eklenir.
                qualities.append(quality)

                # Final en iyi vektorun CSV dosya yolu hazirlanir.
                vector_file = os.path.join(run_dir, 'best_vector_run_%02d.csv' % run_index)

                # Convergence boyunca en iyi vektor gecmisinin CSV dosya yolu hazirlanir.
                history_file = os.path.join(run_dir, 'best_vector_history_run_%02d.csv' % run_index)

                # Convergence grafiginin PNG dosya yolu hazirlanir.
                convergence_file = os.path.join(run_dir, 'convergence_run_%02d.png' % run_index)

                # Final cozum vektoru grafiginin PNG dosya yolu hazirlanir.
                vector_plot_file = os.path.join(run_dir, 'best_vector_run_%02d.png' % run_index)

                # Final best vektor tek satir CSV olarak kaydedilir.
                np.savetxt(vector_file, best.reshape(1, -1), delimiter=',', fmt='%.9f')

                # Best vektor gecmisi, her satir bir kayit ani olacak sekilde CSV'ye yazilir.
                np.savetxt(history_file, np.array(alg.history_vectors), delimiter=',', fmt='%.9f')

                # Objective skorunun zamanla nasil degistigini gosteren convergence grafigi kaydedilir.
                save_convergence_plot(convergence_file, alg, obj_name, alg_name, run_index)

                # Final cozum vektorunun boyutlara gore deger grafigi kaydedilir.
                save_vector_plot(vector_plot_file, best, obj_name, alg_name, run_index)

                # Bu kosuya ait tum onemli bilgiler summary tablosu icin listeye eklenir.
                summary_rows.append([
                    # Benchmark problemi adi.
                    obj_name,

                    # Algoritma adi.
                    alg_name,

                    # Kosu numarasi.
                    run_index,

                    # Final objective skoru.
                    objective_score,

                    # Benchmark quality degeri.
                    quality,

                    # Final vektor CSV dosyasi.
                    vector_file,

                    # Vektor gecmisi CSV dosyasi.
                    history_file,

                    # Convergence PNG dosyasi.
                    convergence_file,

                    # Final vektor grafigi PNG dosyasi.
                    vector_plot_file,
                ])

                # Konsola bu kosunun skor, kalite ve vektor bilgisi yazilir.
                print('%s run %02d score=%.9f quality=%.9f vector=%s' % (
                    # Algoritma adi yazdirilir.
                    alg_name,

                    # Kosu numarasi yazdirilir.
                    run_index,

                    # Objective skoru yazdirilir.
                    objective_score,

                    # Quality degeri yazdirilir.
                    quality,

                    # Vektor okunabilir kisalikta string'e cevrilir.
                    np.array2string(best, precision=4, separator=', ')
                ))

            # Algoritmanin bu problemdeki tekrar kosularindan ozet istatistikleri yazdirilir.
            print('%s median=%.9f mean=%.9f min=%.9f max=%.9f variance=%.9f' % (
                # Algoritma adi yazdirilir.
                alg_name,

                # Quality degerlerinin medyani yazdirilir.
                np.median(qualities),

                # Quality degerlerinin ortalamasi yazdirilir.
                np.mean(qualities),

                # En kucuk quality degeri yazdirilir.
                np.min(qualities),

                # En buyuk quality degeri yazdirilir.
                np.max(qualities),

                # Quality degerlerinin varyansi yazdirilir.
                np.var(qualities)
            ))

    # Tum kosularin ozet CSV dosya yolu hazirlanir.
    summary_path = os.path.join(out_dir, 'summary.csv')

    # summary.csv yazma modunda acilir.
    with open(summary_path, 'w', newline='') as f:
        # CSV writer nesnesi olusturulur.
        writer = csv.writer(f)

        # CSV baslik satiri yazilir.
        writer.writerow([
            # Problem adi sutunu.
            'objective',

            # Algoritma adi sutunu.
            'algorithm',

            # Kosu numarasi sutunu.
            'run',

            # Objective score sutunu.
            'objective_score',

            # Quality sutunu.
            'quality',

            # Final vektor dosyasi sutunu.
            'best_vector_file',

            # Vektor gecmisi dosyasi sutunu.
            'best_vector_history_file',

            # Convergence grafigi dosyasi sutunu.
            'convergence_plot',

            # Final vektor grafigi dosyasi sutunu.
            'best_vector_plot',
        ])

        # Bellekte biriken tum kosu satirlari CSV'ye yazilir.
        writer.writerows(summary_rows)

    # Summary grafiginde x ekseni etiketleri burada tutulur.
    labels = []

    # Summary grafiginde y ekseni degerleri burada tutulur.
    values = []

    # Her problem tekrar gezilir.
    for obj in objs:
        # Her algoritma tekrar gezilir.
        for alg_name, _ in algs:
            # Bu problem-algoritma ikilisine ait quality degerleri summary satirlarindan secilir.
            selected = [float(row[4]) for row in summary_rows if row[0] == obj.name() and row[1] == alg_name]

            # X ekseni etiketi problem adi ve algoritma adindan olusur.
            labels.append('%s\n%s' % (obj.name(), alg_name))

            # Y eksenine tekrar kosularinin medyan quality degeri yazilir.
            values.append(np.median(selected))

    # Ozet karsilastirma grafigi icin yeni figur acilir.
    plt.figure(figsize=(12, 6))

    # Her problem-algoritma ikilisi icin bir bar cizilir.
    plt.bar(range(len(values)), values)

    # X ekseni etiketleri okunabilir olsun diye egimli yazilir.
    plt.xticks(range(len(values)), labels, rotation=35, ha='right')

    # Y ekseni etiketi medyan quality olarak ayarlanir.
    plt.ylabel('Median quality')

    # Grafik basligi Firefly karsilastirma ozeti olarak ayarlanir.
    plt.title('Firefly comparison summary')

    # Sadece y ekseninde hafif izgara acilir.
    plt.grid(True, axis='y', alpha=0.3)

    # Etiketler kesilmesin diye yerlesim sikilastirilir.
    plt.tight_layout()

    # Summary quality grafigi ana cikti klasorune kaydedilir.
    plt.savefig(os.path.join(out_dir, 'summary_quality.png'), dpi=150)

    # Figur kapatilir.
    plt.close()

    # Ciktilarin hangi klasore yazildigi konsola bildirilir.
    print('\nOutputs written to:', os.path.abspath(out_dir))
