#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QProcess>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPixmap>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void importCziFile();
    void saveImage();

private:
    QWidget *centralWidget;
    QVBoxLayout *mainLayout;
    QPushButton *importButton;
    QPushButton *saveButton;
    QLabel *imageLabel;
};

#endif // MAINWINDOW_H
