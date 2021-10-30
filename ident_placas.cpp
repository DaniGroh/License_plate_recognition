#include "iostream"
#include "string"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/ml/ml.hpp>


using namespace std;
using namespace cv;

const Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
const int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
const int ADAPTIVE_THRESH_WEIGHT = 9;
const Scalar SCALAR_BLACK = Scalar(0.0, 0.0, 0.0);
const Scalar SCALAR_WHITE = Scalar(255.0, 255.0, 255.0);
const Scalar SCALAR_YELLOW = Scalar(0.0, 255.0, 255.0);
const Scalar SCALAR_GREEN = Scalar(0.0, 255.0, 0.0);
const Scalar SCALAR_RED = Scalar(0.0, 0.0, 255.0);

const int MIN_PIXEL_WIDTH = 2;
const int MIN_PIXEL_HEIGHT = 8;

const double MIN_ASPECT_RATIO = 0.25;
const double MAX_ASPECT_RATIO = 1.0;

const int MIN_PIXEL_AREA = 80;

const double MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3;
const double MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0;

const double MAX_CHANGE_IN_AREA = 0.5;

const double MAX_CHANGE_IN_WIDTH = 0.8;
const double MAX_CHANGE_IN_HEIGHT = 0.2;

const double MAX_ANGLE_BETWEEN_CHARS = 20.0;

const int MIN_NUMBER_OF_MATCHING_CHARS = 3;

const int RESIZED_CHAR_IMAGE_WIDTH = 20;
const int RESIZED_CHAR_IMAGE_HEIGHT = 30;

const int MIN_CONTOUR_AREA = 500;

const string aux1 = "placa";
const string aux2 = ".jpg";

struct Img
{
    string nomearquivo;
    Mat Imagem;
    Mat imgOriginal;
    Mat imgHSV;
    Mat imgCinza;
    vector<Mat> vectorOfHSVImages;
    Mat imgValue;
    Mat imgBlurred;
    Mat imgThresh;

    vector<Point> contornos;
    vector<Point> contornos2;

};

void exibir_imagem(Mat imagem, string nome, int y, int x )
{
    namedWindow(nome, WINDOW_NORMAL);
    resizeWindow(nome,200,200);
    moveWindow(nome,y,x);
    imshow(nome,imagem);
}

int main( int argc, char** argv )
{
    Img Imagens;

    const char* NomeArquivo = argc >= 2 ? argv[1] : "imagens/placa5.jpg";

    Imagens.imgOriginal = imread(NomeArquivo);
    /*for(int i = 0; i < 8; ++i)
    {
    	nomearquivo = aux1 + char(i) + aux2;

    	//const char* NomeArquivo = argc >= 2 ? argv[1] : Imagens.nomearquivo;
    	Imagens.push_back(imread(nomearquivo));
    }*/

    if(Imagens.imgOriginal.empty())
    {
        cout<<"Imagem "<< Imagens.nomearquivo << "Inexistente";
        return -1;
    }

    Rect boundingRect;
    Rect boundingRect1;
    Rect boundingRect2;

    int intCenterX;
    int intCenterY;

    double Diagonal_Size;
    double Aspect_Ratio;

    Imagens.Imagem = Mat::zeros(255, 255, CV_8UC3);

    exibir_imagem(Imagens.imgOriginal,"Imagem_Original",200,100);

    cvtColor(Imagens.imgOriginal, Imagens.imgHSV, CV_BGR2HSV);

    exibir_imagem(Imagens.imgHSV,"Imagem_HSV",1300,500);

    split(Imagens.imgOriginal, Imagens.vectorOfHSVImages);
    Imagens.imgValue = Imagens.vectorOfHSVImages[2];
    Imagens.imgCinza = Imagens.imgValue;

    exibir_imagem(Imagens.imgCinza,"Imagem_Cinza",1000,100);

    //cinza
    Mat imgTopHat;
    Mat imgBlackHat;
    Mat imgGrayscalePlusTopHat;
    Mat imgGrayscalePlusTopHatMinusBlackHat;

    Mat structuringElement = getStructuringElement(CV_SHAPE_RECT, Size(3, 3));

    morphologyEx(Imagens.imgCinza, imgTopHat, CV_MOP_TOPHAT, structuringElement);
    morphologyEx(Imagens.imgCinza, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

    imgGrayscalePlusTopHat = Imagens.imgCinza + imgTopHat;
    Imagens.imgCinza = imgGrayscalePlusTopHat - imgBlackHat;

    exibir_imagem(Imagens.imgCinza,"Imagem_Cinza2",1000,100);
    ///

    GaussianBlur(Imagens.imgCinza, Imagens.imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);

    exibir_imagem(Imagens.imgBlurred,"Imagem_Blurred",200,500);

    adaptiveThreshold(Imagens.imgBlurred, Imagens.imgThresh, 255.0, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);

    exibir_imagem(Imagens.imgThresh,"Imagem_Thresholding",600,500);

    vector<vector<Point> > contours;
    vector<vector<Point> > contornos_de_char_p1;
    vector<vector<Point> > contornos_de_char_p2;
    vector<vector<Point> > contornos_de_char_p3;
    vector<vector<Point> > contornos_de_char_p4;

    vector<Vec4i> hierarchy;

    findContours( Imagens.imgThresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


    Mat drawing = Mat::zeros( Imagens.imgThresh.size(), CV_8UC3 );
    Mat imagem_contornos_char_p1 = Mat::zeros( Imagens.imgThresh.size(), CV_8UC3 );
    Mat imagem_contornos_char_p2 = Mat::zeros( Imagens.imgThresh.size(), CV_8UC3 );
    Mat imagem_contornos_char_p3 = Mat::zeros( Imagens.imgThresh.size(), CV_8UC3 );
    Mat imagem_contornos_char_p4 = Mat::zeros( Imagens.imgThresh.size(), CV_8UC3 );



    for( int i = 0; i< contours.size(); i++ )
    {
        Imagens.contornos = contours[i];

        boundingRect = cv::boundingRect(Imagens.contornos);

        intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
        intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

        Diagonal_Size = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));

        Aspect_Ratio = (float)boundingRect.width / (float)boundingRect.height;

        int NumeroPossiveisChar = 0;

        if(boundingRect.area() > MIN_PIXEL_AREA &&
                boundingRect.width > MIN_PIXEL_WIDTH &&
                boundingRect.height > MIN_PIXEL_HEIGHT &&
                MIN_ASPECT_RATIO < Aspect_Ratio &&
                Aspect_Ratio < MAX_ASPECT_RATIO)
        {
            NumeroPossiveisChar++;
            contornos_de_char_p1.push_back(Imagens.contornos);
            drawContours( drawing, contours, i, SCALAR_WHITE);
            // rectangle( drawing,boundingRect.tl(), boundingRect.br(),SCALAR_GREEN , 2, 8, 0 );
        }
    }

    exibir_imagem(drawing,"contornos_refinado",1000,500);

    for( int i = 0; i< contornos_de_char_p1.size(); i++ )
    {
        drawContours( imagem_contornos_char_p1, contornos_de_char_p1, i, SCALAR_RED);
    }

    exibir_imagem(imagem_contornos_char_p1,"Imagem_Char_p1",1300,500);

    double CentroX1;
    double CentroY1;
    double CentroX2;
    double CentroY2;

    double distancia_entre_chars_eixo_X;
    double distancia_entre_chars_eixo_Y;
    double distancia;
    double angulo_entre_chars;
    int tot = 0;
    double Adj = 0;
    double opst = 0;
    double ang_rad = 0;
    double ang_deg = 0;
    double area_aceitavel = 0;
    double largura_aceitavel = 0;
    double altura_aceitavel = 0;
    vector<double> XMIN;
    vector<double> XMAX;
    vector<double> YMIN;
    vector<double> YMAX;

    for( int i = 0; i< contornos_de_char_p1.size(); i++ )
    {
        Imagens.contornos = contornos_de_char_p1[i];

        boundingRect1 = cv::boundingRect(Imagens.contornos);

        CentroX1 = (boundingRect1.x + boundingRect1.x + boundingRect1.width) / 2;
        CentroY1 = (boundingRect1.y + boundingRect1.y + boundingRect1.height) / 2;

        for( int j = 0; j< contornos_de_char_p1.size(); j++ )
        {
            Imagens.contornos2 = contornos_de_char_p1[j];

            boundingRect2 = cv::boundingRect(Imagens.contornos2);

            CentroX2 = (boundingRect2.x + boundingRect2.x + boundingRect2.width) / 2;
            CentroY2 = (boundingRect2.y + boundingRect2.y + boundingRect2.height) / 2;
            Diagonal_Size = sqrt(pow(boundingRect2.width, 2) + pow(boundingRect2.height, 2));

            distancia_entre_chars_eixo_X = abs(CentroX1 - CentroX2);
            distancia_entre_chars_eixo_Y = abs(CentroY1 - CentroY2);

            distancia = sqrt(pow(distancia_entre_chars_eixo_X,2)+(distancia_entre_chars_eixo_Y,2));
            ang_rad = atan(distancia_entre_chars_eixo_Y/distancia_entre_chars_eixo_X);
            ang_deg = ang_rad * (180.0 * CV_PI);

            area_aceitavel = ((double)(abs((boundingRect1.area()/boundingRect2.area())))/(double)(abs(boundingRect1.area())));
            largura_aceitavel = ((double)(abs(boundingRect2.width - boundingRect1.width))/(double)(abs(boundingRect1.width)));
            altura_aceitavel = ((double)(abs(boundingRect2.height - boundingRect1.height))/(double)(abs(boundingRect1.height)));


            //cout<<"Total Contornos :"<<contornos_de_char_p1.size()<<"  Distancia :"<<distancia<< "  Total :"<<tot<<endl;

            if((distancia < Diagonal_Size * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
                    (ang_deg<MAX_ANGLE_BETWEEN_CHARS)&&(area_aceitavel<0.5) &&
                    (altura_aceitavel < MAX_CHANGE_IN_HEIGHT)&&
                    (largura_aceitavel < MAX_CHANGE_IN_WIDTH))
            {
                contornos_de_char_p3.push_back(Imagens.contornos2);
                tot = tot+1;
            }
        }
    }
    cout<<"P3 SIZE 1 :"<<contornos_de_char_p3.size()<<endl;

    for(int i = 0; i < contornos_de_char_p3.size(); i ++)
    {
        Imagens.contornos = contornos_de_char_p3[i];

        boundingRect = cv::boundingRect(Imagens.contornos);

        for(int j = i+1; j < contornos_de_char_p3.size(); j ++)
        {

            Imagens.contornos2 = contornos_de_char_p3[j];

            boundingRect2 = cv::boundingRect(Imagens.contornos2);

            if((boundingRect.x == boundingRect2.x)&&(boundingRect.y == boundingRect2.y))
            {
                //  if(boundingRect2.x < (boundingRect.x + boundingRect.width)&&(boundingRect2.y<(boundingRect.y+boundingRect.height)))
                //{
                contornos_de_char_p3.erase(contornos_de_char_p3.begin() + j);
                //}
            }
        }
    }

    double menor = 100000000000000;
    int aux;
    double valor;
    int conta;

    //ORDENAÇÃO EM RELAÇÃO AO EIXO X, MENOR PARA MAIOR;

    cout<<"P3 SIZE :"<<contornos_de_char_p3.size()<<endl;

    while(!(contornos_de_char_p3.empty()))
    {
        menor = 10000000000000;
        aux = 0;
        for(int i = 0; i < contornos_de_char_p3.size(); i ++)
        {
            Imagens.contornos = contornos_de_char_p3[i];
            conta++;
            boundingRect = cv::boundingRect(Imagens.contornos);

            valor = boundingRect.x;

            if(valor < menor)
            {
                menor = valor;
                //cout<<"Menor: "<<menor<<"Conta :"<<conta<<endl;
                aux = i;
            }
        }
        Imagens.contornos = contornos_de_char_p3[aux];
        contornos_de_char_p4.push_back(Imagens.contornos);
        contornos_de_char_p3.erase(contornos_de_char_p3.begin() + aux);
        cout<<"menor"<<menor<<"indice"<<aux<<endl;
    }
    cout<<"\nP4 SIZE :"<<contornos_de_char_p4.size()<<endl;

    for( int i = 0; i< contornos_de_char_p4.size(); i++ )
    {
        Imagens.contornos = contornos_de_char_p4[i];

        boundingRect = cv::boundingRect(Imagens.contornos);

        drawContours( imagem_contornos_char_p4, contornos_de_char_p4, i, SCALAR_RED);
        rectangle( imagem_contornos_char_p4,boundingRect.tl(), boundingRect.br(),SCALAR_GREEN , 2, 8, 0 );
    }

    exibir_imagem(imagem_contornos_char_p4,"Imagem_Char_p4",1000,500);

    double prox = 0;
    int   comb = 0;

    for(int i = 0; i < contornos_de_char_p4.size(); i ++)
    {
        Imagens.contornos = contornos_de_char_p4[i];

        boundingRect = cv::boundingRect(Imagens.contornos);
        comb = 0;

        for(int j = i+1; j < contornos_de_char_p4.size(); j ++)
        {
            Imagens.contornos2 = contornos_de_char_p4[j];

            boundingRect2 = cv::boundingRect(Imagens.contornos2);

            double difAlt = (double)(abs(boundingRect2.y - boundingRect.y))/(double)(abs(boundingRect.height));

            if(((boundingRect2.x) > (boundingRect.x + boundingRect.width))&& (difAlt < 0.2))
            {
                comb ++;
                if(comb >=7)
                {
                    comb =0;
                    XMIN.push_back(boundingRect.x);
                    XMAX.push_back(boundingRect2.x + boundingRect2.width);
                    YMIN.push_back(boundingRect.y);
                    YMAX.push_back(boundingRect2.y + boundingRect2.height);
                }
            }
        }

        cout<< "Combinacoes   :" << comb << endl;
    }

    for(int i = 0; i < YMAX.size(); i++)
    {
        double altura = (YMAX[i] - YMIN[i]);
        double largura = (XMAX[i] - XMIN[i]);

        if(altura < (2*(largura/7)) && (largura > MIN_PIXEL_WIDTH*7) && (altura > (MIN_PIXEL_HEIGHT*3)))
        {

            Rect reg_placa((XMIN[i]),(YMIN[i]),largura,altura);
            /*
            cout<<"altura :"<<altura<<endl;
            cout<<"largura :" <<largura<<endl;
            cout<< "xmin :"<<XMIN[i]-20<<endl;
            cout<< "ymin :"<<YMIN[i]-10<<endl;
            */
            Mat croppedImage = Imagens.imgOriginal(reg_placa);

            char palavra[50];
            palavra[0]=i+'0';

            exibir_imagem(croppedImage,palavra,950,500);
        }
    }

    waitKey(0);

    return 0;
}
