#include "mainwindow.h"
#include <QMessageBox>  // Add this include to fix the error

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {
    // Central widget and layout
    centralWidget = new QWidget(this);
    mainLayout = new QVBoxLayout(centralWidget);

    // "Import CZI file" button
    importButton = new QPushButton("Import czi file (.czi)", this);
    mainLayout->addWidget(importButton);

    // Image display label
    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(imageLabel);

    // "Save Image" button (disabled by default)
    saveButton = new QPushButton("Save Image", this);
    saveButton->setEnabled(false);
    mainLayout->addWidget(saveButton);

    // Connect signals to slots
    connect(importButton, &QPushButton::clicked, this, &MainWindow::importCziFile);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveImage);

    // Set the central widget and layout
    setCentralWidget(centralWidget);
}

void MainWindow::importCziFile() {
    QString filePath = QFileDialog::getOpenFileName(this, "Select CZI File", "", "CZI Files (*.czi)");
    if (!filePath.isEmpty()) {
        // Start the Python process
        QProcess pythonProcess;

        // Connect to capture the standard output and error
        connect(&pythonProcess, &QProcess::readyReadStandardOutput, this, [&]() {
            QByteArray output = pythonProcess.readAllStandardOutput();
            qDebug() << "Python Output: " << output;
        });
        connect(&pythonProcess, &QProcess::readyReadStandardError, this, [&]() {
            QByteArray error = pythonProcess.readAllStandardError();
            qDebug() << "Python Error: " << error;
        });

        // Start the process
        pythonProcess.start("C:/Users/Administrator/AppData/Local/Programs/Python/Python39/python.exe", QStringList() << "process_czi.py" << filePath);

        // Wait for the process to finish
        pythonProcess.waitForFinished();

        // Check if the output image exists
        QString outputImagePath = "output_image.png";
        QFile outputFile(outputImagePath);
        if (!outputFile.exists()) {
            QMessageBox::warning(this, "Error", "The processed image file does not exist.");
            return;
        }

        // Load the processed image
        QPixmap pixmap(outputImagePath);
        if (!pixmap.isNull()) {
            imageLabel->setPixmap(pixmap.scaled(800, 800, Qt::KeepAspectRatio));
            saveButton->setEnabled(true);  // Enable Save button
        } else {
            QMessageBox::warning(this, "Error", "Failed to load the processed image.");
        }
    }
}




void MainWindow::saveImage() {
    QString savePath = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG Files (*.png)");
    if (!savePath.isEmpty() && !imageLabel->pixmap().isNull()) {
        imageLabel->pixmap().save(savePath);
    }
}


