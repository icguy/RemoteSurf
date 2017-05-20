A Tárgymanipuláció kulcspontok detektálásán alapuló képfeldolgozással című diplomatervhez tartozó program rövidített felhasználói útmutatója.
Készítette: Ayhan Dániel

A program jelenleg (a diplomaterv leadásakor, 2017 május) működőképes az IIT tanszék robotlaborjában lévő számítógépen.

Robotprogram: RSTEST.prg
A PLC program a laborban lévő számítógépen van rajta, a Dokumentumok/AD/Project mappában.
A PC program indításához a ModbusGUIsimple.py szkriptet kell indítani.

A rendszer bekapcsolása a következőképpen történik: 
1. A PLC és a robotkar vezérlőjének elindítása
2. Ha szükséges, a PLC újraprogramozása, a PLC és robot áramtalanítása, majd GOTO 1.
3. A PC program indítása (és a Connect gombbal catlakozás)
4. Home pozíció beállítása (Home gombra kattintás, majd Set gombok bármelyikének megnyomása)
5. Robotprogram indítása

A kéz-szem kalibráció minden mérés elején elvégzendő a Calibrate gombra kattintva (és a felugró ablak egyetlen gombját megnyomva). Így csak a képek készülnek el, a kalibráció nem fut le.
A képek elkészítése után opcionálisan a kamerakalibráció is lefuttatható az opencv_calib/opencv_calib.py futtatásával, miután az 51. sorban beállítottuk az előző futtatás kimeneti mappáját (out/YYYY_MM_DD_hh_mm_ss formátumú mappa).
A kéz-szem kalibráció elvégzéséhez az arrangement_calib/chessboard_test.py-t kell futtatni, miután az előzőleg ismertetett mappát megadtuk neki a 419. sorban.
Ez után a ModbusGUIsimple.py szkriptben a 27. sorban ugyanezt a mappát kell beállítani, és a korábban generált pontfelhő alapján a Find gombbal már lehet indítani az objektum keresését. Miután lefutott a keresés, a GUI-n frissülnek a célpozíció értékei. Odaküldeni a robotot a Set gombok bármelyikének megnyomásával lehet.

Saját pontfelhő generálásához az SFMSolver.py szkript calc_data_from_files kezdetű függvényeit lehet használni, de ez sok manuális beállítást igényel, nem automatikus, főleg a képek külső paramétereinek meghatározása. Én a calc_data_from_files_tiarng_simple függvényt használtam a saját pontfelhő generálásához.

A futtatás sajnos nem lehetséges bizonyos preprocesszált fájlok nélkül, amelyeket nem tudtam feltölteni, mert a melléklet meghaladta volna a maximális méretet. Ezeket a fájlokat a https://github.com/icguy/RemoteSurf repositoryból lehet elérni, a "cache" mappát, és az out/2017_3_8__14_51_22 mappát kell letölteni. 