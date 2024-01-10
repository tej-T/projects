CREATE TABLE PROJECT1_BOOK (
    ISBN13 NUMBER,
    TITLE  VARCHAR2(4000),
    CONSTRAINT BOOKPK PRIMARY KEY ( ISBN13 )
);

CREATE TABLE PROJECT1_BOOK_AUTHORS (
    AUTHOR_ID VARCHAR2(4000),
    ISBN13    NUMBER,
    CONSTRAINT BAPK PRIMARY KEY ( AUTHOR_ID ),
    CONSTRAINT BAFK FOREIGN KEY ( ISBN13 )
        REFERENCES PROJECT1_BOOK
);

CREATE TABLE PROJECT1_AUTHORS (
    AUTHOR_ID VARCHAR2(4000),
    NAME      VARCHAR2(4000),
    CONSTRAINT AUTHPK PRIMARY KEY ( AUTHOR_ID ),
    CONSTRAINT AUTHFK FOREIGN KEY ( AUTHOR_ID )
        REFERENCES PROJECT1_BOOK_AUTHORS
);

CREATE TABLE PROJECT1_LIBRARY_BRANCH (
    BRANCH_ID   NUMBER,
    BRANCH_NAME VARCHAR2(4000),
    ADDRESS     VARCHAR2(4000),
    CONSTRAINT LBPK PRIMARY KEY ( BRANCH_ID )
);

CREATE TABLE PROJECT1_BOOK_COPIES (
    BOOK_ID   VARCHAR2(4000),
    ISBN13    NUMBER,
    BRANCH_ID NUMBER,
    CONSTRAINT BCPK PRIMARY KEY ( BOOK_ID ),
    CONSTRAINT BCFK FOREIGN KEY ( ISBN13 )
        REFERENCES PROJECT1_BOOK,
    CONSTRAINT BCFK2 FOREIGN KEY ( BRANCH_ID )
        REFERENCES PROJECT1_LIBRARY_BRANCH
);

CREATE TABLE PROJECT1_BORROWER (
    CARD_NO VARCHAR2(4000),
    SSN     VARCHAR2(4000),
    FNAME   VARCHAR2(4000),
    LNAME   VARCHAR2(4000),
    ADDRESS VARCHAR2(4000),
    PHONE   VARCHAR2(4000),
    CONSTRAINT BORPK PRIMARY KEY ( CARD_NO )
);

CREATE TABLE PROJECT1_BOOK_LOANS (
    LOAN_ID  VARCHAR2(4000),
    BOOK_ID  VARCHAR2(4000),
    CARD_NO  VARCHAR2(4000),
    DATE_OUT DATE,
    DUE_DATE DATE,
    DATE_IN  DATE,
    CONSTRAINT BLPK PRIMARY KEY ( LOAN_ID ),
    CONSTRAINT BLFK1 FOREIGN KEY ( BOOK_ID )
        REFERENCES PROJECT1_BOOK_COPIES,
    CONSTRAINT BLFK2 FOREIGN KEY ( CARD_NO )
        REFERENCES PROJECT1_BORROWER
);

CREATE TABLE PROJECT1_FINES (
    LOAN_ID  VARCHAR2(4000),
    FINE_AMT FLOAT,
    PAID     FLOAT,
    CONSTRAINT FINEPK PRIMARY KEY ( LOAN_ID ),
    CONSTRAINT FINEFK FOREIGN KEY ( LOAN_ID )
        REFERENCES PROJECT1_BOOK_LOANS
);
