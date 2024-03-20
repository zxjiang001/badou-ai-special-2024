FROM harbor.changdu.ltd/public/java17-jdk:v2 AS build-env
WORKDIR /springboot

COPY . ./

RUN mvn clean package -DskipTests=true

FROM harbor.changdu.ltd/public/java17:v1
WORKDIR /ABExperimentServer
COPY --from=build-env /springboot/target /ABExperimentServer

#设置字符编码
ENV LANG en_US.UTF-8

CMD ["java", "-XX:+UseContainerSupport", "-XX:InitialRAMPercentage=80.0", "-XX:MinRAMPercentage=80.0", "-XX:MaxRAMPercentage=80.0", "-XX:-UseAdaptiveSizePolicy", "-jar", "ABExperimentServer.jar", "--server.port=80"]