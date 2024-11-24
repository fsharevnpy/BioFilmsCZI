#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainWindow window;
    window.setWindowTitle("CZI Viewer");
    window.resize(900, 900);
    window.show();

    return app.exec();
}
