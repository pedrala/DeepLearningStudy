
![serviceServer_client](https://github.com/pedrala/DeepLearningStudy/blob/main/img/serviceServer_client.png?raw=true)

![serviceDiagram](https://github.com/pedrala/DeepLearningStudy/blob/main/img/serviceDiagram.png?raw=true)

![client_server](https://github.com/pedrala/DeepLearningStudy/blob/main/img/client_server.png?raw=true)

operator.py
------------
구글의 오픈 이미지 데이터세트로 모델을 학습시킬 수 있는 파이썬코드
여기에서 cctv사물인지를 위해 내려받은 데이터셋트를 SSD MobileNet v1 네트워크로 학습시키는 함수를 확인할 수 있음

```python

import random  # 랜덤 숫자 생성을 위한 모듈

from ros_study_msgs.srv import ArithmeticOperator  # ArithmeticOperator 서비스 메시지 임포트
import rclpy  # ROS 2 파이썬 클라이언트 라이브러리
from rclpy.node import Node  # ROS 2 노드 클래스 임포트

# Operator 클래스 정의: 서비스 클라이언트로서 ArithmeticOperator 서비스를 호출
class Operator(Node):

    def __init__(self):
        super().__init__('operator')  # 'operator'라는 이름으로 노드 초기화

        # ArithmeticOperator 서비스를 위한 클라이언트 생성
        self.arithmetic_service_client = self.create_client(
            ArithmeticOperator,  # 사용할 서비스 타입
            'arithmetic_operator')  # 서비스 이름

        # 서비스가 준비될 때까지 기다리는 루프
        while not self.arithmetic_service_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warning('The arithmetic_operator service not available.')  # 서비스가 준비되지 않으면 경고 로그 출력

    # 서비스 요청을 보내는 함수
    def send_request(self):
        service_request = ArithmeticOperator.Request()  # 서비스 요청 메시지 생성
        service_request.arithmetic_operator = random.randint(1, 4)  # 랜덤하게 1부터 4까지의 연산자 선택 (예: 더하기, 빼기, 곱하기, 나누기)
        futures = self.arithmetic_service_client.call_async(service_request)  # 비동기로 서비스 호출
        return futures  # 서비스 응답을 대기하는 future 객체 반환

# 메인 함수 정의: 노드를 실행하고 서비스 요청을 보내고 응답을 처리
def main(args=None):
    rclpy.init(args=args)  # rclpy 초기화
    operator = Operator()  # Operator 클래스 인스턴스 생성
    future = operator.send_request()  # 첫 번째 서비스 요청 보내기
    user_trigger = True  # 사용자 트리거 변수로, 사용자가 새로운 요청을 할 준비가 되었는지 관리

    try:
        while rclpy.ok():  # rclpy가 정상적으로 실행되는 동안 루프를 계속 돌림
            if user_trigger is True:  # 사용자가 요청을 할 준비가 되었을 때
                rclpy.spin_once(operator)  # 노드의 콜백 함수들을 한 번만 실행 (블록하지 않음)
                if future.done():  # future가 완료되었는지 확인
                    try:
                        service_response = future.result()  # 서비스 응답 결과를 받음
                    except Exception as e:  # 서비스 호출에 실패한 경우 예외 처리
                        operator.get_logger().warn('Service call failed: {}'.format(str(e)))  # 오류 메시지 출력
                    else:
                        operator.get_logger().info(
                            'Result: {}'.format(service_response.arithmetic_result))  # 성공하면 결과 출력
                        user_trigger = False  # 다음 입력을 기다리기 위해 트리거를 False로 변경
            else:
                input('Press Enter for next service call.')  # 다음 서비스 요청을 위해 Enter 입력 대기
                future = operator.send_request()  # 새 요청 보내기
                user_trigger = True  # 트리거를 True로 설정하여 요청 준비 완료

    except KeyboardInterrupt:  # 사용자가 Ctrl+C로 프로그램을 종료할 때
        operator.get_logger().info('Keyboard Interrupt (SIGINT)')  # 로그에 인터럽트 메시지 출력

    operator.destroy_node()  # 노드 종료
    rclpy.shutdown()  # rclpy 종료

# 만약 이 스크립트가 메인 스크립트로 실행되면 main() 함수 실행
if __name__ == '__main__':
    main()

```
calculator.py
------------

```python

        # ArithmeticOperator 서비스를 제공
        self.arithmetic_service_server = self.create_service(
            ArithmeticOperator,  # 제공할 서비스 타입
            'arithmetic_operator',  # 서비스 이름
            self.get_arithmetic_operator,  # 서비스 호출 시 실행될 콜백 함수
            callback_group=self.callback_group)  # 콜백 그룹 사용
```
            
calculator 노드는 서브스크라이브하여 저장하고 있는 변수 a와 b와 operator 노드로부터 요청 값으로 받은 연산자를 이용하여 계산(a 연산자 b)하고 operator 노드에게 연산의 결괏값을 서비스 응답값으로 보낸다.

이 중 서비스 서버와 관련한 코드는 아래와 같다. 서버 관련 코드는 서비스 서버로 선언하는 부분과 콜백함수를 지정하는 것이다. arithmetic_service_server이 서비스 서버로 이는 Node 클래스의 create_service 함수를 이용하여 서비스 서버로 선언되었으며 서비스의 타입으로 ArithmeticOperator으로 선언하였고, 서비스 이름으로는 'arithmetic_operator', 서비스 클라이언트로부터 서비스 요청이 있으면 실행되는 콜백함수는 get_arithmetic_operator 으로 지정했으며 멀티 스레드 병렬 콜백함수 실행을 위해 지난번 강좌에서 설명한 callback_group 설정을 하였다. 

이러한 설정들은 서비스 서버를 위한 기본 설정이고 실제 서비스 요청에 해당되는 특정 수행 코드가 수행되는 부분은 get_arithmetic_operator 이라는 콜백함수임을 알아두자.

```python
    # 서비스 요청을 처리하는 콜백 함수
    def get_arithmetic_operator(self, request, response):
        self.argument_operator = request.arithmetic_operator  # 서비스로 받은 연산자 값 저장
        # 계산 결과를 계산하는 함수 호출
        self.argument_result = self.calculate_given_formula(
            self.argument_a,  # 첫 번째 값
            self.argument_b,  # 두 번째 값
            self.argument_operator)  # 연산자 값
        # 서비스 응답에 계산 결과 저장
        response.arithmetic_result = self.argument_result
        # 연산 공식을 문자열로 저장
        self.argument_formula = '{0} {1} {2} = {3}'.format(
            self.argument_a,
            self.operator[self.argument_operator - 1],  # 연산자 기호를 리스트에서 가져옴
            self.argument_b,
            self.argument_result)
        # 계산 결과 공식 로그 출력
        self.get_logger().info(self.argument_formula)
        return response  # 서비스 응답 반환

```

get_arithmetic_operator 함수의 내용을 보도록 하자. 우선 제일 먼저 request와 response 이라는 매개 변수가 보이는데 이는 ArithmeticOperator() 클래스로 생성된 인터페이스로 서비스 요청에 해당되는 request 부분과 응답에 해당되는 response으로 구분된다. 

get_arithmetic_operator 함수는 서비스 요청이 있을 때 실행되는 콜백함수인데 여기서는 서비스 요청시 요청값으로 받은 연산자와 이전에 토픽 서브스크라이버가 토픽 값으로 전달받아 저장해둔 변수 a, b를 전달받은 연산자로 연산 후에 결괏값을 서비스 응답값으로 반환한다. 

이 함수의 첫 줄에서 request.arithmetic_operator를 받아와서 calculate_given_formula 함수에서 서비스 요청값으로 받은 연산자에 따라 연산하는 코드를 볼 수 있을 것이다. calculate_given_formula 함수로부터 받은 연산 결괏값은 response.arithmetic_result에 저장하고 끝으로 관련 수식을 문자열로 표현하여 get_logger().info() 함수를 통해 화면에 표시하고 있다. 

```python

    # 연산을 수행하는 함수
    def calculate_given_formula(self, a, b, operator):
        if operator == ArithmeticOperator.Request.PLUS:
            self.argument_result = a + b  # 더하기
        elif operator == ArithmeticOperator.Request.MINUS:
            self.argument_result = a - b  # 빼기
        elif operator == ArithmeticOperator.Request.MULTIPLY:
            self.argument_result = a * b  # 곱하기
        elif operator == ArithmeticOperator.Request.DIVISION:
            try:
                self.argument_result = a / b  # 나누기
            except ZeroDivisionError:
                self.get_logger().error('ZeroDivisionError!')  # 0으로 나눌 때 오류 로그 출력
                self.argument_result = 0.0  # 오류 발생 시 결과를 0으로 설정
                return self.argument_result
        else:
            self.get_logger().error('Please make sure arithmetic operator(plus, minus, multiply, division).')  # 잘못된 연산자 오류 로그
            self.argument_result = 0.0  # 오류 발생 시 결과를 0으로 설정
        return self.argument_result  # 연산 결과 반환

```

calculate_given_formula 함수로 인수 a, b를 가지고 주어진 연산자 operator에 따라 사칙연산을 하여 결괏값을 반환한다.


```python
import rclpy
from rclpy.executors import MultiThreadedExecutor

from topic_service_action_rclpy_example.calculator.calculator import Calculator


def main(args=None):
    rclpy.init(args=args)
    try:
        calculator = Calculator()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(calculator)
        try:
            executor.spin()
        except KeyboardInterrupt:
            calculator.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            executor.shutdown()
            calculator.arithmetic_action_server.destroy()
            calculator.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

```
