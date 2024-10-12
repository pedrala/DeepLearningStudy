
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
