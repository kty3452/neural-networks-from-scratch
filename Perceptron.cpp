#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int n_features; // 가중치 벡터 길이(특징 개수와 같음)
    double *w; // 가중치 벡터 
    double b; // 바이어스
    double lr; // 학습률
} Perceptron;

// --유틸--

/* 내적 */
static double dot(const double *a, const double *b, int n){
    double s = 0.0;
    for(int i = 0;i < n;i++) s += a[i] * b[i];
    return s;
}

/* 계단 함수 */
static int sign(double z){ 
    return (z >= 0.0) ? +1 : -1;
}

/* 순서를 섞는 함수 */
static void shuffle_indices(int *idx, int n){
    for(int i = n - 1;i > 0;i--){
        int j = rand() % (i + 1);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

// --퍼셉트론--

/* perceptron 초기화 */
void perceptron_init(Perceptron *p, int n_features, double lr){
    p->n_features = n_features;                      // 특성(입력 차원 수) 저장
    p->w = (double*)calloc(n_features, sizeof(double)); // 가중치 배열을 0으로 초기화하여 동적 할당
    p->b = 0.0;                                      // 바이어스(bias) 초기화
    p->lr = lr;                                      // 학습률(learning rate) 저장
    if(!p->w){                                       // 메모리 할당 실패 체크
        fprintf(stderr, "메모리 할당 실패\n");
        exit(1);
    }
}

/* perceptron 메모리 해제 */
void perceptron_free(Perceptron *p){
    if(p->w) free(p->w);
    p->w = NULL;
}

/* 활성 함수용 함수형 포인터*/
typedef int (*ActivationFunc)(double);

/* perceptron 예측 */
int perceptron_predict_one(const Perceptron *p, const double *x, ActivationFunc activation){
    double z = dot(p->w, x, p->n_features) + p->b;
    return activation(z);
}

/* perceptron 학습 */
void perceptron_fit(
    Perceptron *p,
    const double *X, // shape: n_sample x n_features 
    const int *y, // label: (-1, +1) -> 이진 분류의 경우
    int n_samples,
    int n_features,
    int epochs,
    ActivationFunc activation,
    int verbose
){
    int *idx = (int*)malloc(sizeof(int) * n_samples);
    if(!idx){ 
        fprintf(stderr, "메모리 할당 실패\n"); 
        exit(1); 
    }
    for(int i = 0;i < n_samples;i++) idx[i] = i; // 샘플 번호 저장

    for(int ep = 1; ep <= epochs; ep++){
        shuffle_indices(idx, n_samples); 

        int errors = 0; // 틀린 횟수 기록
        for(int t = 0; t < n_samples; t++){
            int i = idx[t];
            const double *xi = &X[i * n_features]; // i번째 샘플의 특징들 추출
            int yi = y[i];

            double z = dot(p->w, xi, n_features) + p->b;
            int yhat = activation(z);

            if(yhat != yi){
                // 오분류시 로젠블랫 업데이트
                for(int k = 0;k < n_features;k++) p->w[k] += p->lr * yi * xi[k];
                p->b += p->lr * yi;
                errors++;
            }
        }
        // 모니터링
        if(verbose) printf("[epoch %d] error=%d\n", ep, errors);

        if(errors == 0){
            // 완벽 분류 달성 시 일찍 종료 (선형분리 가능이면 유용)
            printf("조기 종료: epoch %d\n", ep);
            break;
        }
    }
    free(idx);
}

void perceptron_print(const Perceptron *p){
    printf("weights = [");
    for(int i = 0;i < p->n_features;i++){
        printf("%s%.6f", (i? ", ":""), p->w[i]);
    }
    printf("], bias = %.6f\n", p->b);
}

int main(){
    srand((unsigned int)time(NULL));

    // 예시: AND 문제 (선형분리 가능)
    // 입력 2차원, label은 {+1, -1}
    // (0, 0) -> -1, (0, 1) -> -1, (1, 0) -> -1, (1, 1) -> +1

    double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    int y[] = { -1, -1, -1, +1 };

    int n_samples = 4;
    int n_features = 2;

    Perceptron p;
    perceptron_init(&p, n_features, /*lr=*/0.1);

    perceptron_fit(&p, X, y, n_samples, n_features, /*epochs=*/10, sign, /*verbose=*/1);

    perceptron_print(&p);

    // 예측
    for(int i = 0;i < n_samples;i++){
        int pred = perceptron_predict_one(&p, &X[i * n_features], sign);
        printf("x=(%.1f, %.1f)  y=%2d  pred=%2d\n",
            X[i*n_features+0], X[i*n_features+1], y[i], pred);
    }

    perceptron_free(&p);

    return 0;

}