
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
calculator 노드는 서브스크라이브하여 저장하고 있는 변수 a와 b와 operator 노드로부터 요청 값으로 받은 연산자를 이용하여 계산(a 연산자 b)하고 operator 노드에게 연산의 결괏값을 서비스 응답값으로 보낸다.

이 중 서비스 서버와 관련한 코드는 아래와 같다. 서버 관련 코드는 서비스 서버로 선언하는 부분과 콜백함수를 지정하는 것이다. arithmetic_service_server이 서비스 서버로 이는 Node 클래스의 create_service 함수를 이용하여 서비스 서버로 선언되었으며 서비스의 타입으로 ArithmeticOperator으로 선언하였고, 서비스 이름으로는 'arithmetic_operator', 서비스 클라이언트로부터 서비스 요청이 있으면 실행되는 콜백함수는 get_arithmetic_operator 으로 지정했으며 멀티 스레드 병렬 콜백함수 실행을 위해 지난번 강좌에서 설명한 callback_group 설정을 하였다. 

이러한 설정들은 서비스 서버를 위한 기본 설정이고 실제 서비스 요청에 해당되는 특정 수행 코드가 수행되는 부분은 get_arithmetic_operator 이라는 콜백함수임을 알아두자.

```python

import time  # 시간 관련 함수 사용을 위한 모듈

from ros_study_msgs.action import ArithmeticChecker  # ArithmeticChecker 액션 메시지 임포트
from ros_study_msgs.msg import ArithmeticArgument  # ArithmeticArgument 메시지 임포트
from ros_study_msgs.srv import ArithmeticOperator  # ArithmeticOperator 서비스 메시지 임포트
from rclpy.action import ActionServer  # ROS 2 액션 서버 클래스 임포트
from rclpy.callback_groups import ReentrantCallbackGroup  # 콜백 그룹 관리 클래스
from rclpy.node import Node  # ROS 2 노드 클래스 임포트
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # QoS 관련 설정

# Calculator 클래스 정의: 메시지 구독, 서비스 제공, 액션 서버 포함
class Calculator(Node):

    def __init__(self):
        super().__init__('calculator')  # 'calculator'라는 이름으로 노드 초기화
        self.argument_a = 0.0  # 구독할 첫 번째 값
        self.argument_b = 0.0  # 구독할 두 번째 값
        self.argument_operator = 0  # 서비스로 받을 연산자 값
        self.argument_result = 0.0  # 연산 결과 값
        self.argument_formula = ''  # 연산 공식을 저장할 문자열
        self.operator = ['+', '-', '*', '/']  # 연산자 리스트 (더하기, 빼기, 곱하기, 나누기)
        self.callback_group = ReentrantCallbackGroup()  # 동시 콜백 처리를 위한 콜백 그룹

        # QoS 설정
        self.declare_parameter('qos_depth', 10)  # 'qos_depth'라는 파라미터 선언
        qos_depth = self.get_parameter('qos_depth').value  # 선언한 파라미터의 값 불러오기

        # QoS 프로파일 생성: 신뢰성, 히스토리, 버퍼 깊이, 내구성 설정
        QOS_RKL10V = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability=QoSDurabilityPolicy.VOLATILE)

        # ArithmeticArgument 메시지 구독 설정
        self.arithmetic_argument_subscriber = self.create_subscription(
            ArithmeticArgument,  # 구독할 메시지 타입
            'arithmetic_argument',  # 토픽 이름
            self.get_arithmetic_argument,  # 메시지 수신 시 호출될 콜백 함수
            QOS_RKL10V,  # QoS 프로파일 적용
            callback_group=self.callback_group)  # 콜백 그룹 사용

        # ArithmeticOperator 서비스를 제공
        self.arithmetic_service_server = self.create_service(
            ArithmeticOperator,  # 제공할 서비스 타입
            'arithmetic_operator',  # 서비스 이름
            self.get_arithmetic_operator,  # 서비스 호출 시 실행될 콜백 함수
            callback_group=self.callback_group)  # 콜백 그룹 사용

        # ArithmeticChecker 액션 서버 설정
        self.arithmetic_action_server = ActionServer(
            self,  # 노드 인스턴스
            ArithmeticChecker,  # 액션 메시지 타입
            'arithmetic_checker',  # 액션 서버 이름
            self.execute_checker,  # 액션 요청 시 실행될 콜백 함수
            callback_group=self.callback_group)  # 콜백 그룹 사용

    # 구독한 메시지의 데이터를 처리하는 콜백 함수
    def get_arithmetic_argument(self, msg):
        self.argument_a = msg.argument_a  # 메시지로 받은 첫 번째 값 저장
        self.argument_b = msg.argument_b  # 메시지로 받은 두 번째 값 저장
        # 메시지의 타임스탬프와 두 수를 로그로 출력
        self.get_logger().info('Timestamp of the message: {0}'.format(msg.stamp))
        self.get_logger().info('Subscribed argument a: {0}'.format(self.argument_a))
        self.get_logger().info('Subscribed argument b: {0}'.format(self.argument_b))

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

    # 액션 서버 요청을 처리하는 함수
    def execute_checker(self, goal_handle):
        self.get_logger().info('Execute arithmetic_checker action!')  # 액션 실행 로그 출력
        feedback_msg = ArithmeticChecker.Feedback()  # 피드백 메시지 생성
        feedback_msg.formula = []  # 공식을 저장할 리스트 초기화
        total_sum = 0.0  # 총합 초기화
        goal_sum = goal_handle.request.goal_sum  # 목표 합계를 클라이언트 요청에서 가져옴
        # 총합이 목표 합계에 도달할 때까지 반복
        while total_sum < goal_sum:
            total_sum += self.argument_result  # 현재 연산 결과를 총합에 더함
            feedback_msg.formula.append(self.argument_formula)  # 공식을 피드백 메시지에 추가
            self.get_logger().info('Feedback: {0}'.format(feedback_msg.formula))  # 피드백 로그 출력
            goal_handle.publish_feedback(feedback_msg)  # 클라이언트에 피드백 전송
            time.sleep(1)  # 1초 대기
        goal_handle.succeed()  # 목표 달성 시 성공 상태 설정
        result = ArithmeticChecker.Result()  # 결과 메시지 생성
        result.all_formula = feedback_msg.formula  # 모든 공식 결과 저장
        result.total_sum = total_sum  # 총합 저장
        return result  # 결과 반환

```
